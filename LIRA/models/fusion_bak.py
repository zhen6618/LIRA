import torch
import torch.nn as nn
import trimesh
from skimage import measure
import numpy as np
from matplotlib import cm
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from collections import Counter
from torchsparse.tensor import PointTensor
from neu_utils import sparse_to_dense_channel, sparse_to_dense_torch
from .modules import ConvGRU
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from copy import deepcopy
import torch.nn.functional as F
from .mask3dformer import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP, Conv3d
from .voxel_position_encoding import PositionEmbeddingCoordsSine
from datasets.visualization import label2mesh
from .modules import keep_largest_connected_region_3d

def compute_3d_hbb(points):
    "points: [N, 3]"
    # Step 1: 计算每个坐标轴的最小值和最大值
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    
    # Step 2: 计算中心点 (bounding box的中心)
    center = (min_vals + max_vals) / 2
    
    # Step 3: 计算边界框的尺寸
    l, w, h = max_vals - min_vals
    
    return np.array([np.around(center[0], decimals=0), np.around(center[1], decimals=0), np.around(center[2], decimals=0),
                     np.around(l, decimals=0), np.around(w, decimals=0), np.around(h, decimals=0)])

def compute_3d_obb(points):
    "points: [N, 3]"
    # Step 1: 计算中心点
    center = np.mean(points, axis=0)

    # Step 2: 通过PCA分析点云的主要方向
    pca = PCA(n_components=3)
    pca.fit(points)
    
    # 主成分，用于定义物体的朝向
    rotation_matrix = pca.components_
    
    # Step 3: 将点云转换到PCA坐标系
    transformed_points = points - center
    transformed_points = transformed_points @ rotation_matrix.T
    
    # Step 4: 计算最小包围盒的尺寸
    min_vals = np.min(transformed_points, axis=0)
    max_vals = np.max(transformed_points, axis=0)
    
    # 长宽高分别为最大值与最小值的差
    l, w, h = max_vals - min_vals
    
    # Step 5: 计算旋转角（在XY平面上的旋转）
    theta = np.arctan2(rotation_matrix[0, 1], rotation_matrix[0, 0])

    return np.array([np.around(center[0], decimals=0), np.around(center[1], decimals=0), np.around(center[2], decimals=0), 
                    np.around(l, decimals=0), np.around(w, decimals=0), np.around(h, decimals=0), np.around(theta, decimals=3)])

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)

    return loss.sum() / num_masks  

def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks

class Fusion(nn.Module):
    """
    Two functionalities of this class:
    1. GRU Fusion module as in the paper. Update hidden state features with ConvGRU.
    2. Substitute TSDF in the global volume when direct_substitute = True.
    """

    def __init__(self, cfg, ch_in=None, direct_substitute=False, contrastive_hidden_dim=128, refine_hidden_dim=256, contrastive_num_layers=3, refine_num_layers=3):
        super(Fusion, self).__init__()
        self.cfg = cfg
        # replace tsdf in global tsdf volume by direct substitute corresponding voxels
        self.direct_substitude = direct_substitute

        if direct_substitute:
            # tsdf
            self.ch_in = [1, 1, 1]
            self.feat_init = 1
        else:
            # features
            self.ch_in = ch_in
            self.feat_init = 0

        self.n_scales = len(cfg.THRESHOLDS) - 1
        self.scene_name = [None, None, None]
        self.instruction_id = [None, None, None]
        self.global_origin = [None, None, None]
        self.target_tsdf_volume = [None, None, None]

        "实例融合"
        self.global_instance = [None]  # instance_id
        self.global_ins_infos = {}

        "网络定义"
        self.num_heads = 8

        self.pos_enc_type = "fourier"
        
        if self.pos_enc_type == "fourier":
            self.contrastive_pos_enc = PositionEmbeddingCoordsSine(
                pos_type="fourier",
                d_pos=contrastive_hidden_dim,
                gauss_scale=1.0,
                normalize=True,
            )
        elif self.pos_enc_type == "sine":
            self.contrastive_pos_enc = PositionEmbeddingCoordsSine(
                pos_type="sine",
                d_pos=contrastive_hidden_dim,
                normalize=True,
            )
        else:
            assert False, "pos enc type not known"

        if self.pos_enc_type == "fourier":
            self.refine_pos_enc = PositionEmbeddingCoordsSine(
                pos_type="fourier",
                d_pos=refine_hidden_dim,
                gauss_scale=1.0,
                normalize=True,
            )
        elif self.pos_enc_type == "sine":
            self.refine_pos_enc = PositionEmbeddingCoordsSine(
                pos_type="sine",
                d_pos=refine_hidden_dim,
                normalize=True,
            )
        else:
            assert False, "pos enc type not known"

        self.text_transform = MLP(input_dim=4096, hidden_dim=contrastive_hidden_dim*2, output_dim=contrastive_hidden_dim, num_layers=2) 

        "*****************************************************   contrastive learning   *****************************************************"
        self.query_feat = nn.Embedding(1, contrastive_hidden_dim)  # learnable query features
        # self.query_embed = nn.Embedding(50, hidden_dim)  # 位置编码 learnable query p.e. 假设最多有50个query
        # self.text_embed = nn.Embedding(50, hidden_dim)  

        self.contrastive_occ_pre_fc = nn.Linear(256, int(contrastive_hidden_dim/2))
        self.contrastive_occ_pre = Conv3d(input_dim=int(contrastive_hidden_dim/2), hidden_dim=int(contrastive_hidden_dim/2), output_dim=int(contrastive_hidden_dim/2), num_layers=12)  
        self.contrastive_occ_pre_fc_2 = nn.Linear(int(contrastive_hidden_dim/2), contrastive_hidden_dim)

        self.contrastive_num_layers = contrastive_num_layers

        self.contrastive_cross_attn_to_occ_feats = nn.ModuleList()
        self.contrastive_cross_attn_to_text_feats = nn.ModuleList()
        self.contrastive_ffn_layers = nn.ModuleList()

        for _ in range(self.contrastive_num_layers):  # Each stage's transformer block loops num_layers times 
            self.contrastive_cross_attn_to_occ_feats.append(
                CrossAttentionLayer(
                    d_model=contrastive_hidden_dim,
                    nhead=self.num_heads,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.contrastive_cross_attn_to_text_feats.append(
                CrossAttentionLayer(
                    d_model=contrastive_hidden_dim,
                    nhead=self.num_heads,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.contrastive_ffn_layers.append(
                FFNLayer(
                    d_model=contrastive_hidden_dim,
                    dim_feedforward=4*contrastive_hidden_dim,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

        self.contrastive_transform = MLP(input_dim=contrastive_hidden_dim, hidden_dim=contrastive_hidden_dim*2, output_dim=contrastive_hidden_dim, num_layers=3)
        self.confidence_transform = MLP(input_dim=contrastive_hidden_dim, hidden_dim=contrastive_hidden_dim, output_dim=1, num_layers=3)

        "*****************************************************   contrastive learning   *****************************************************"
        "*****************************************************   3D Occ Mask Refinement   *****************************************************"
        # # self.refine_transform = MLP(input_dim=refine_hidden_dim, hidden_dim=refine_hidden_dim*2, output_dim=refine_hidden_dim, num_layers=2)

        # self.query_feat_refine = nn.Embedding(1, refine_hidden_dim)  # learnable query features
        # self.query_embed_refine = nn.Embedding(30, refine_hidden_dim)  # 位置编码 learnable query p.e. 假设最多有30个query

        # self.refine_occ_pre_fc = nn.Linear(256, refine_hidden_dim)
        # self.refine_occ_pre = Conv3d(input_dim=refine_hidden_dim, hidden_dim=refine_hidden_dim, output_dim=refine_hidden_dim, num_layers=5)
        # self.mask_feat_refine = Conv3d(input_dim=refine_hidden_dim, hidden_dim=refine_hidden_dim, output_dim=refine_hidden_dim, num_layers=5)

        # self.refine_num_layers = refine_num_layers
        # self.refine_self_attention_layers = nn.ModuleList()
        # self.refine_cross_attention_layers = nn.ModuleList()
        # self.refine_ffn_layers = nn.ModuleList()

        # for _ in range(self.refine_num_layers):  # Each stage's transformer block loops num_layers times
        #     self.refine_self_attention_layers.append(
        #         SelfAttentionLayer(
        #             d_model=refine_hidden_dim,
        #             nhead=self.num_heads,
        #             dropout=0.0,
        #             normalize_before=False,
        #         )
        #     )

        #     self.refine_cross_attention_layers.append(
        #         CrossAttentionLayer(
        #             d_model=refine_hidden_dim,
        #             nhead=self.num_heads,
        #             dropout=0.0,
        #             normalize_before=False,
        #         )
        #     )

        #     self.refine_ffn_layers.append(
        #         FFNLayer(
        #             d_model=refine_hidden_dim,
        #             dim_feedforward=4*refine_hidden_dim,
        #             dropout=0.0,
        #             normalize_before=False,
        #         )
        #     )

        # self.decoder_norm = nn.LayerNorm(refine_hidden_dim)
        # self.mask_embed = MLP(refine_hidden_dim, refine_hidden_dim*4, refine_hidden_dim, 3)  # pred mask

        "*****************************************************   3D Occ Mask Refinement   *****************************************************"

        "*****************************************************   LLM   *****************************************************"

        # self.LLM_path = "/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/LISA_checkpoint/Qwen-7B-Chat"

        # self.LLM_tokenizer = AutoTokenizer.from_pretrained(self.LLM_path, trust_remote_code=True)

        # self.LLM_model = AutoModelForCausalLM.from_pretrained(
        #     self.LLM_path,
        #     device_map="auto",
        #     trust_remote_code=True,
        #     bf16=True,
        # ).eval()

        # self.LLM_model.generation_config = GenerationConfig.from_pretrained(self.LLM_path, trust_remote_code=True)

        # for param in self.LLM_model.parameters():
        #     param.requires_grad = False

        # # for name, param in self.LLM_model.named_parameters():
        # #     if param.grad is not None:
        # #         print(f"Parameter {name} has gradient: {param.grad}")

        "*****************************************************   LLM   *****************************************************"

    def get_pos_encs(self, coords, layer, spitial_shape=None):  # [N, 3]
        device = coords.device

        scene_min = coords.min(dim=0)[0]  # 考虑到与global实例特征进行融合，occ的位置编码对 实例级归一化的位置坐标 进行编码
        scene_max = coords.max(dim=0)[0]

        scene_min = scene_min.view(-1, 3)
        scene_max = scene_max.view(-1, 3)

        pos_encodings_pcd = layer(coords[None, ...].float(), input_range=[scene_min, scene_max])  # [1, c, N]

        return pos_encodings_pcd
    
    def contrastive_iou(self, A, B):
        """
        计算布尔张量 A[M, N] 和 B[D, N] 之间的 IoU(Intersection over Union)
        """
        # 计算交集 (Intersection) 和并集 (Union)
        intersection = (A.unsqueeze(1) & B.unsqueeze(0)).sum(dim=2)  # (M, D)
        union = (A.unsqueeze(1) | B.unsqueeze(0)).sum(dim=2)  # (M, D)

        # 避免除以零，计算 IoU
        iou = intersection.float() / (union.float() + 1e-6)  # (M, D)

        return iou
    
    # def compute_contrastive_loss_all(self, features, labels, temperature=0.07):  # 参考CLIP中temperature=0.07
    #     """
    #     计算基于余弦相似度的对比损失，支持多对多标签匹配
        
    #     features: 特征向量 (n, c), n为实例数, c为特征维度
    #     labels: 每个实例的标签 (n,)，标签表示每个实例属于哪个对象
    #     temperature: 温度系数，用于控制相似度分布

    #     """
    #     # 计算特征之间的余弦相似度
    #     similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)  # [n_ins, n_ins]

    #     # 效仿EmbodedSAM，对每个类别分别计算
    #     contrastive_loss = []
    #     for ins_gt_id in torch.unique(labels):
    #         if ins_gt_id == 1000:
    #             continue
            
    #         label_mask = (labels == ins_gt_id)

    #         label_matrix = label_mask.unsqueeze(1).repeat(1, len(labels)) & label_mask.unsqueeze(0).repeat(len(labels), 1)  # [n_ins, n_ins]
    #         similarity_sum_id = torch.sum(torch.exp(similarity_matrix[label_matrix] / temperature))

    #         label_matrix_all = label_mask.unsqueeze(1).repeat(1, len(labels)) | label_mask.unsqueeze(0).repeat(len(labels), 1)  # [n_ins, n_ins]
    #         similarity_sum_all = torch.sum(torch.exp(similarity_matrix[label_matrix_all] / temperature))

    #         contrastive_loss.append(-torch.log(similarity_sum_id / similarity_sum_all))  # 0 ~ 1
        
    #     if len(contrastive_loss) == 0:
    #         valid_loss = False
    #         print("all 1000")
    #         return features.sum() * 0, valid_loss
    #     else: 
    #         contrastive_loss = sum(contrastive_loss) / len(contrastive_loss)  # loss per instance
    #         valid_loss = True

    #         return contrastive_loss, valid_loss


    def similarity_to_cluster_labels(self, similarity_matrix, threshold=0.5):
        device = similarity_matrix.device

        # Ensure similarity_matrix is numpy array
        similarity_matrix = similarity_matrix.detach().cpu().numpy() 

        # Convert similarity to distance (1 - similarity)
        distance_matrix = 1 - similarity_matrix

        # Use DBSCAN to find clusters
        clustering = DBSCAN(eps=threshold, min_samples=1, metric='precomputed')
        cluster_labels = clustering.fit_predict(distance_matrix)

        return torch.from_numpy(cluster_labels).long().to(device)

    def compute_contrastive_loss_all_ce(self, features, labels):  
        """
        计算基于余弦相似度的对比损失，支持多对多标签匹配
        
        features: 特征向量 (n, c), n为实例数, c为特征维度
        labels: 每个实例的标签 (n,)，标签表示每个实例属于哪个对象

        """
        # 计算特征之间的余弦相似度
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)  # [n_ins, n_ins] [-1, 1]
        similarity_matrix = (similarity_matrix + 1) / 2  # [0, 1] 
        similarity_matrix = torch.clamp(similarity_matrix, 0, 1)

        # TODO 正负样本类别不均衡处理

        # 计算目标分数
        target_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)

        # 计算二元交叉熵损失
        loss = F.binary_cross_entropy(similarity_matrix, target_matrix.float())

        # 推理
        # cluster_labels = self.similarity_to_cluster_labels(similarity_matrix, threshold=0.5)

        return loss, similarity_matrix

    def compute_similarty_matrix(self, features):  
        """
        features: 特征向量 (n, c), n为实例数, c为特征维度
        """
        # 计算特征之间的余弦相似度
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)  # [n_ins, n_ins] [-1, 1]
        similarity_matrix = (similarity_matrix + 1) / 2  # [0, 1] 
        similarity_matrix = torch.clamp(similarity_matrix, 0, 1)

        return similarity_matrix
    
    def compute_global_ins_fusion_loss(self, features, labels, ious=None):  
        """
        计算基于余弦相似度的对比损失，支持多对多标签匹配
        
        features: 特征向量 (n, c), n为实例数, c为特征维度
        labels: 每个实例的标签 (n,)，标签表示每个实例属于哪个对象
        ious: 每个实例的置信度 (n,)

        """
        # 计算特征之间的余弦相似度
        similarity_matrix = self.compute_similarty_matrix(features)

        # TODO 正负样本类别不均衡处理

        # 计算目标分数
        target_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
        # target_ious = ious.unsqueeze(1) * ious.unsqueeze(0)
        # target_matrix = target_matrix * target_ious

        # 计算二元交叉熵损失
        loss = F.binary_cross_entropy(similarity_matrix, target_matrix.float())

        return loss


    # def compute_contrastive_loss_two(self, t_feats, t_1_feats, t_labels, t_1_labels, temperature=0.07):  # 参考CLIP中temperature=0.07
    #         """
    #         计算基于余弦相似度的对比损失
            
    #         t_feats, t_1_feats: 特征向量 (n, c), n为实例数, c为特征维度
    #         t_labels, t_1_labels: 每个实例的标签 (n,)，标签表示每个实例属于哪个对象
    #         temperature: 温度系数，用于控制相似度分布

    #         """
    #         if len(t_labels) == 0 or len(t_1_labels) == 0:
    #             valid_loss = False
    #             return 0, valid_loss

    #         similarity_matrix = F.cosine_similarity(t_feats.unsqueeze(1), t_1_feats.unsqueeze(0), dim=2)  # [n_ins, n_ins]
    #         similarity_matrix = torch.exp(similarity_matrix / temperature)

    #         loss = []

    #         # t ➡️ t+1
    #         for i in range(len(t_labels)):
    #             match_ins = t_1_labels == t_labels[i]
    #             if match_ins.sum() == 1:  # 超过两个以上匹配时，不计算
    #                 numerator = similarity_matrix[i, match_ins]
    #                 denominator = similarity_matrix[i, :].sum()
    #                 loss.append(-torch.log(numerator / denominator))

    #         # t+1 ➡️ t
    #         for j in range(len(t_1_labels)):
    #             match_ins = t_labels == t_1_labels[j]
    #             if match_ins.sum() == 1:
    #                 numerator = similarity_matrix[match_ins, j]
    #                 denominator = similarity_matrix[:, j].sum()
    #                 loss.append(-torch.log(numerator / denominator))

    #         # print("loss: ", loss)
    #         if len(loss) == 0 or sum(loss).item() == 0:
    #             valid_loss = False
    #             return 0, valid_loss
    #         else: 
    #             loss = sum(loss) / len(loss)  # loss per instance
    #             valid_loss = True
    #             return loss, valid_loss

    def compute_refine_loss(self, predictions_mask, gt_masks_refine):
        """
        计算mask损失
        
        refine_out: list: [Q, L]
        gt_masks_refine: [Q, L]

        """        
        dice_weight = 0.1
        mask_weight = 2.0
        num_masks = len(predictions_mask[0])

        refine_loss = 0
        for pred_mask in predictions_mask:
            per_loss_mask = sigmoid_ce_loss(pred_mask, gt_masks_refine, num_masks)  # loss per instance
            per_loss_dice = dice_loss(pred_mask, gt_masks_refine, num_masks)  # loss per instance

            refine_loss += per_loss_mask * mask_weight + per_loss_dice * dice_weight

        return refine_loss

    def get_local_ins_infos(self, ins_info):
        """  img_level:
        对于每张图像, 将每个实例对应一个id:
        list 0: (num_voxels, 3), 每个voxel对应的坐标
        list 1: (num_instance, num_voxels), 每个ins对应的mask, bool类型
        list 2: 
            list 0: dict: {'id': 1, 'category': 'black chair'}
            ...
        list 3: (num_instance, text_feat_dim), 每个实例对应的text feats
        """

        """ instance_info:
        每个batch一个列表, 每个列表内:
        'labels': 实例的类别标签(包含40类语义标签)
        'masks': shape:[num_semantic_labels, num_voxels]
        'ins_ids': 实例的id
        """
        device = ins_info['occ_level'].device

        query_num = 0  # query数量 = ins数量
        for i in range(len(ins_info['img_level'])):
            query_num += len(ins_info['img_level'][i][2])

        # occ feats
        occ_feats_all = ins_info['occ_level']  # [nV, c] 

        # mask
        mask_all = []
        for i in range(len(ins_info['img_level'])):
            if len(ins_info['img_level'][i][1]) > 0:
                mask_all.append(ins_info['img_level'][i][1])
        if len(mask_all) > 0:
            mask_all = torch.cat(mask_all, dim=0)  # [n_ins, nV]

        # text feats
        text_feats = []
        for i in range(len(ins_info['img_level'])):
            if len(ins_info['img_level'][i][3]) > 0:
                text_feats.append(ins_info['img_level'][i][3])
        if len(text_feats) > 0:
            text_feats = torch.cat(text_feats, dim=0)  # [n_ins, nV]
            text_feats = self.text_transform(text_feats)  # 1024 -> 256 [n_ins, c]

        "*****************************************************   没有预测出实例   *****************************************************"
        # if query_num == 0 or 'ins_ids' not in ins_info['gt']:
        if query_num == 0:
            confidence_loss = 0
            valid_confidence_loss = False
            return None, None, None, None, None, None, confidence_loss, valid_confidence_loss
        
        # 获取每个实例的 color, class
        color_cls_all = []
        for i in range(len(ins_info['img_level'])):
            if len(ins_info['img_level'][i][2]) > 0:
                for j in range(len(ins_info['img_level'][i][2])):
                    color_cls_all.append(ins_info['img_level'][i][2][j]['category'])

        # 寻找match GT实例 
        if 'masks' in ins_info['gt']:
            iou_matrix = []
            for iou_i in range(len(mask_all)):
                mask_confine = ins_info['gt']['contrastive_iou_mask'][iou_i]
                current_pred_mask = mask_all[iou_i][mask_confine].unsqueeze(0)
                current_target_mask = ins_info['gt']['masks'][:, mask_confine]
                iou_matrix.append(self.contrastive_iou(current_pred_mask, current_target_mask))
            iou_matrix = torch.cat(iou_matrix, dim=0)

            match_gt_index = torch.argmax(iou_matrix, dim=1)  # [n_ins, ]
            match_gt_iou = iou_matrix[torch.arange(query_num), match_gt_index]  # [n_ins, ]
            match_gt_ins_ids = ins_info['gt']['ins_ids'][match_gt_index]  
            match_gt_ins_ids[match_gt_iou < 0.5] = 1000

        else:  # 如果GT没有候选实例
            match_gt_iou = torch.zeros_like(mask_all[:, 0]).to(torch.float)
            match_gt_ins_ids = torch.ones_like(mask_all[:, 0]).to(torch.long) * 1000

        "*****************************************************   contrastive learning   *****************************************************"
        # QxBxC query
        query_contrastive = self.query_feat.weight.repeat(query_num, 1).unsqueeze(1) 
        # query_contrastive_embed = self.query_embed.weight[:query_num, :].unsqueeze(1)  # 位置编码

        # 只对非空的voxel进行attention，加速计算
        mask_all_no_empty = mask_all.sum(dim=0) > 0
        occ_feats_contrastive = occ_feats_all[mask_all_no_empty]  # [nV-, c]
        mask_contrastive = mask_all[:, mask_all_no_empty]  # [n_ins, nV-]  
        for no_empty_tmp in ins_info['img_level']:
            if len(no_empty_tmp[0]) > 0:
                coords_no_empty = no_empty_tmp[0][mask_all_no_empty]  # [nV-, 3]
                coords_all = no_empty_tmp[0]
                break

        # coords最小归于0
        coords_min = coords_no_empty.min(dim=0)[0] 
        coords_max = coords_no_empty.max(dim=0)[0]
        coords_no_empty = coords_no_empty - coords_min
        contrastive_spitial_shape = coords_max - coords_min
        contrastive_spitial_shape = tuple(contrastive_spitial_shape.tolist())

        # 将occ_feats提前预处理转换一下
        occ_feats_contrastive = self.contrastive_occ_pre_fc(occ_feats_contrastive)
        occ_feats_contrastive = self.contrastive_occ_pre(x=occ_feats_contrastive,
                                                        coords=coords_no_empty,  # [nV-, 3]
                                                        spitial_shape=contrastive_spitial_shape,
                                                        )
        occ_feats_contrastive = self.contrastive_occ_pre_fc_2(occ_feats_contrastive)

        # NxBxC
        occ_feats_contrastive = occ_feats_contrastive.unsqueeze(1)  # [nV-, 1, c]  
        # [Q, L] -> [B, Q, L] -> [B, h, Q, L] -> [B*h, Q, L]
        mask_contrastive = mask_contrastive.unsqueeze(0)  # [1, n_ins, nV-]=[B, Q, L]
        mask_contrastive = mask_contrastive.unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1)
        mask_contrastive = ~mask_contrastive  # False: 允许attention; True: 不允许attention
        mask_contrastive = mask_contrastive.detach()  # TODO 检查所有梯度的反向传播信息应该有还是没有

        # text mask
        text_mask = torch.ones(query_num, query_num, dtype=torch.bool, device=device)
        text_mask[torch.arange(query_num), torch.arange(query_num)] = False
        # [Q, L] -> [B, Q, L] -> [B, h, Q, L] -> [B*h, Q, L]
        text_mask = text_mask.unsqueeze(0).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1)
        text_mask = text_mask.detach()
        # NxBxC
        text_feats_contrastive = text_feats.unsqueeze(1)  # [n_ins, 1, c]
        # pos_text = self.text_embed.weight[:query_num, :].unsqueeze(1)  # 位置编码

        pos_occ = self.get_pos_encs(coords_no_empty, self.contrastive_pos_enc)
        # flatten BxCxN to NxBxC
        pos_occ = pos_occ.permute(2, 0, 1)

        for j in range(self.contrastive_num_layers):  # transformer loop
            # cross-attention to occ_feats
            query_contrastive = self.contrastive_cross_attn_to_occ_feats[j](
                query_contrastive, occ_feats_contrastive,
                memory_mask=mask_contrastive,  # For a binary mask, a True value indicates that the corresponding position is not allowed to attend
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos_occ,
            )

            # cross-attention to text_feats
            query_contrastive = self.contrastive_cross_attn_to_text_feats[j](
                query_contrastive, text_feats_contrastive,
                memory_mask=text_mask,  # For a binary mask, a True value indicates that the corresponding position is not allowed to attend
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
            )

            # FFN
            query_contrastive = self.contrastive_ffn_layers[j](
                query_contrastive
            )

        # MLP转换到对比损失空间 和 置信度空间
        query_contrastive_transform = self.contrastive_transform(query_contrastive.squeeze(1))  # [n_ins, c]  
        query_confidence_transform = self.confidence_transform(query_contrastive.squeeze(1))  # [n_ins, 1]

        # # ---  compute contrastive loss  ---
        # # 去除1000的ins_id，减少contrastive loss损失干扰
        # valid_ins_ids = match_gt_ins_ids != 1000
        # if len(valid_ins_ids) < 0:
        #     valid_loss = False
        #     contrastive_loss = 0
        #     similarity_matrix = None
        #     print("all 1000")

        # else:
        #     valid_query_contrastive_transform = query_contrastive_transform[valid_ins_ids]
        #     valid_match_gt_ins_ids = match_gt_ins_ids[valid_ins_ids]

        #     if len(torch.unique(valid_match_gt_ins_ids)) > 1:  # 286
        #         print("exist negative sample: ", valid_match_gt_ins_ids)

        #     "***************************   9帧一起算对比损失   ***************************"
        #     if len(valid_match_gt_ins_ids) <= 1:
        #         contrastive_loss = 0
        #         valid_loss = False
        #         similarity_matrix = None
        #     else:
        #         contrastive_loss, similarity_matrix = self.compute_contrastive_loss_all_ce(valid_query_contrastive_transform, valid_match_gt_ins_ids)
        #         valid_loss = True

        "计算置信度损失"
        query_confidence_transform = query_confidence_transform.squeeze(1).sigmoid()  
        query_confidence_transform = torch.clamp(query_confidence_transform, 0, 1)

        match_gt_iou = torch.clamp(match_gt_iou, 0, 1)

        # if sum(match_gt_ins_ids == 1000) > 0:
        #     print("confidence 1000")
        confidence_loss = F.mse_loss(query_confidence_transform, match_gt_iou)
        valid_confidence_loss = True

        "invalid=1000 舍弃"
        valid_fusion = match_gt_ins_ids != 1000
        if sum(valid_fusion) > 0:
            query_contrastive_transform = query_contrastive_transform[valid_fusion]
            mask_all = mask_all[valid_fusion]
            match_gt_iou = match_gt_iou[valid_fusion]
            match_gt_ins_ids = match_gt_ins_ids[valid_fusion]
            color_cls_all = [a for a, b in zip(color_cls_all, valid_fusion) if b]
        else:
            query_contrastive_transform, mask_all, match_gt_iou, match_gt_ins_ids, coords_all, color_cls_all = None, None, None, None, None, None


        return query_contrastive_transform, mask_all, match_gt_iou, match_gt_ins_ids, coords_all, color_cls_all, confidence_loss, valid_confidence_loss


    def update_tsdf_map(self, coords_target, tsdf_target, relative_origin, scale):

        global_tsdf_target = self.target_tsdf_volume[scale].F
        global_coords_target = self.target_tsdf_volume[scale].C

        dim = (torch.Tensor(self.cfg.N_VOX).cuda() // 2 ** (self.cfg.N_LAYER - scale - 1)).int()
        dim_list = dim.data.cpu().numpy().tolist()

        global_coords_target = global_coords_target - relative_origin
        valid_target = ((global_coords_target < dim) & (global_coords_target >= 0)).all(dim=-1)
        valid_coords_target  = global_coords_target[valid_target]

        # combine current tsdf and global tsdf
        coords_target = torch.cat([global_coords_target[valid_target], coords_target])[:, :3]
        tsdf_target = torch.cat([global_tsdf_target[valid_target], tsdf_target.unsqueeze(-1)])
        # sparse to dense
        target_volume = sparse_to_dense_channel(coords_target, tsdf_target, dim_list, 1, 1, tsdf_target.device)

        # update
        target_volume = target_volume.squeeze()
        volume_index_target = target_volume.abs() < 1
        self.target_tsdf_volume[scale].F = torch.cat([self.target_tsdf_volume[scale].F[valid_target == False], target_volume[target_volume.abs() < 1].unsqueeze(-1)])
        target_coords = torch.nonzero(target_volume.abs() < 1) + relative_origin

        self.target_tsdf_volume[scale].C = torch.cat([self.target_tsdf_volume[scale].C[valid_target == False], target_coords])

        return valid_target, valid_coords_target, volume_index_target


    def global_fusion(self, query_local, mask_local, match_iou_local, match_ins_ids_local, coords_local, color_cls_local, scale, relative_origin, 
                      valid_target, valid_coords_target, volume_index_target):
        """
        query_local: [N, c]
        mask_local: [N, nV]
        match_iou_local: [N, ]
        match_ins_ids_local: [N, ]
        coords_local: [N, 3]
        color_cls_local: list: [N,] 

        self.global_instance: [nV, ] ins_id
        self.global_ins_infos:
            'ins_ids 1'
                {
                    'query': [c]  # 特征
                    'ious': [1, ]  # confidence
                    'match_count': [1, ]  # 累计匹配次数
                    'target_ins_ids': list  # 目标ins_id
                }
            'ins_ids 2'
                {
                    'query': [c]
                    'ious': [1, ]
                    'match_count': [1, ]
                    'target_ins_ids': list
                }
            ...
        """
        device = volume_index_target.device
        fused_instance_ids = torch.zeros(volume_index_target.sum(), dtype=self.global_instance.dtype, device=device)  # 初始化实例标号 [nV', ]

        # 统一FBV内的occ
        dim = (torch.Tensor(self.cfg.N_VOX).cuda() // 2 ** (self.cfg.N_LAYER - scale - 1)).int()
        dim_list = dim.data.cpu().numpy().tolist()

        ins_global = self.global_instance[valid_target]
        ins_volume_global = sparse_to_dense_channel(valid_coords_target, ins_global.unsqueeze(-1), dim_list, 1, 0, device)  # [96, 96, 96, 1]
        ins_global = ins_volume_global[volume_index_target].squeeze(1)  # [nV']

        # local没有预测出实例，保持history
        if query_local is None:
            fused_instance_ids = ins_global
            self.global_instance = torch.cat([self.global_instance[valid_target == False], fused_instance_ids])  # 更新历史实例volume
            contrastive_loss = 0
            valid_loss = False
            return contrastive_loss, valid_loss

        mask_local = mask_local.transpose(1, 0)  # [nV, N] 
        ins_volume_local = sparse_to_dense_channel(coords_local, mask_local, dim_list, mask_local.shape[1], 0, device)  # [96, 96, 96, N]
        mask_local = ins_volume_local[volume_index_target].transpose(1, 0)  # [N, nV']

        # # 寻找历史voxel与当前voxel重合的那部分
        # coords_global = self.target_tsdf_volume[scale].C  
        # coords_global = coords_global - relative_origin  # 转换到局部坐标系
        # valid_global = ((coords_global < dim) & (coords_global >= 0)).all(dim=-1)

        # coords_tmp_global = coords_global[valid_global] 
        # valid_tmp_global = self.voxel_match_fast(coords_tmp_global, coords_local)
        # valid_global[valid_global.clone()] = valid_tmp_global.to(torch.bool).to(device)

        # 历史实例信息
        if len(self.global_instance) > 0:
            max_ins_id_global = int(torch.max(self.global_instance).cpu().item())
        else:
            max_ins_id_global = 0

        unique_ins_global = torch.unique(ins_global).to(torch.int)

        query_global = []
        target_ins_ids_global = []
        ious_global = []
        mask_global = []
        color_cls_global = []
        for uni_ins_global in unique_ins_global:
            if uni_ins_global == 0:  # 背景实例id=0
                continue
            query_global.append(self.global_ins_infos[str(uni_ins_global.item())]['query'].unsqueeze(0))
            ious_global.append(self.global_ins_infos[str(uni_ins_global.item())]['ious'].unsqueeze(0))  
            mask_global.append((ins_global == uni_ins_global).unsqueeze(0))

            ins_ids_counter = Counter(self.global_ins_infos[str(uni_ins_global.item())]['target_ins_ids'])
            target_ins_ids_global.append(ins_ids_counter.most_common(1)[0][0]) 

            color_cls_counter = Counter(self.global_ins_infos[str(uni_ins_global.item())]['color_cls'])
            color_cls_global.append(color_cls_counter.most_common(1)[0][0]) 

        if len(query_global) > 0:
            query_global = torch.cat(query_global, dim=0)
            target_ins_ids_global = torch.tensor(target_ins_ids_global).to(match_ins_ids_local.dtype).to(device)
            ious_global = torch.cat(ious_global, dim=0)
            mask_global = torch.cat(mask_global, dim=0)

            query = torch.cat([query_global, query_local], dim=0)  # [M+N, c]
            target_ins_ids = torch.cat([target_ins_ids_global, match_ins_ids_local], dim=0)  # [M+N, ]
            ious = torch.cat([ious_global, match_iou_local], dim=0)  # [M+N, ]
            masks = torch.cat([mask_global, mask_local], dim=0)  # [M+N, nV]
            color_clses = color_cls_global + color_cls_local
        else:
            query = query_local
            target_ins_ids = match_ins_ids_local
            ious = match_iou_local
            masks = mask_local
            color_clses = color_cls_local

        # 指示哪些位置是global (注意global总是排在local前面)
        is_global = torch.zeros_like(target_ins_ids).to(torch.bool)
        is_global[:len(target_ins_ids_global)] = True

        'local and global fusion for training loss'
        # # 只与非1000实例进行损失计算
        # is_valid_target_ins_ids = target_ins_ids != 1000
        # valid_target_ins_ids = target_ins_ids[is_valid_target_ins_ids]
        # valid_query = query[is_valid_target_ins_ids]

        "only local for training loss"
        # 只与非1000实例进行损失计算
        is_valid_target_ins_ids = match_ins_ids_local != 1000
        valid_target_ins_ids = match_ins_ids_local[is_valid_target_ins_ids]
        valid_query = query_local[is_valid_target_ins_ids]

        # 计算损失
        if len(valid_query) > 1:
            # if len(torch.unique(valid_target_ins_ids)) > 1 and len(query_global) > 0:
            if len(torch.unique(valid_target_ins_ids)) > 1:
                print("len(torch.unique(valid_target_ins_ids)) > 1") 
            contrastive_loss = self.compute_global_ins_fusion_loss(features=valid_query, labels=valid_target_ins_ids)
            valid_loss = True
        else:
            contrastive_loss = 0
            valid_loss = False      
        
        # 实例匹配与融合(包括1000)
        similarity_matrix = self.compute_similarty_matrix(query)
        match_ids = self.similarity_to_cluster_labels(similarity_matrix, threshold=0.5)  # 转换成距离矩阵后，应取小值

        with torch.no_grad():
            unique_match_ids, unique_counts = torch.unique(match_ids, return_counts=True)
            _, unique_sorted_indices = torch.sort(unique_counts, descending=True)  # 对出现次数降序排序，优先对匹配次数多的进行融合，这样有助于剔除invalid实例的mask区域
            unique_match_ids = unique_match_ids[unique_sorted_indices]

            for match_id in unique_match_ids:
                match_i = match_ids == match_id
                match_global = match_i & is_global

                cur_mask = masks[match_i].any(dim=0)  # [nV, ]
                match_cls = [a for a, b in zip(color_clses, match_i) if b]

                if sum(match_global) > 0:  # 继承历史实例标号
                    global_idx = torch.where(match_global == True)[0][0]
                    assign_ins_id = unique_ins_global[1:][global_idx].item()

                    # 更新实例id
                    fused_instance_ids[(fused_instance_ids.clone() == 0) & cur_mask] = assign_ins_id  
                    # 更新query
                    self.global_ins_infos[str(assign_ins_id)]['query'] = self.global_ins_infos[str(assign_ins_id)]['query'] * self.global_ins_infos[str(assign_ins_id)]['match_count'] + query[match_i][1:].sum(0)
                    self.global_ins_infos[str(assign_ins_id)]['query'] = self.global_ins_infos[str(assign_ins_id)]['query'] / (self.global_ins_infos[str(assign_ins_id)]['match_count'] + match_i.sum() - 1)
                    # 更新ious
                    self.global_ins_infos[str(assign_ins_id)]['ious'] = self.global_ins_infos[str(assign_ins_id)]['ious'] * self.global_ins_infos[str(assign_ins_id)]['match_count'] + ious[match_i][1:].sum(0)
                    self.global_ins_infos[str(assign_ins_id)]['ious'] = self.global_ins_infos[str(assign_ins_id)]['ious'] / (self.global_ins_infos[str(assign_ins_id)]['match_count'] + match_i.sum() - 1)
                    # 更新match_count
                    self.global_ins_infos[str(assign_ins_id)]['match_count'] = self.global_ins_infos[str(assign_ins_id)]['match_count'] + match_i.sum() - 1
                    # 更新target ins ids   
                    self.global_ins_infos[str(assign_ins_id)]['target_ins_ids'] = self.global_ins_infos[str(assign_ins_id)]['target_ins_ids'] + target_ins_ids[match_i][1:].tolist()   
                    # 更新color class
                    self.global_ins_infos[str(assign_ins_id)]['color_cls'] = self.global_ins_infos[str(assign_ins_id)]['color_cls'] + match_cls[1:]   

                else:  # 新建实例标号

                    # 更新实例id
                    fused_instance_ids[(fused_instance_ids.clone() == 0) & cur_mask] = max_ins_id_global + 1
                    max_ins_id_global += 1
                    
                    self.global_ins_infos[str(max_ins_id_global)] = {'query': query[match_i].sum(0) / match_i.sum(),
                                                                    'ious': ious[match_i].sum(0) / match_i.sum(),
                                                                    'match_count': match_i.sum(),
                                                                    'target_ins_ids': target_ins_ids[match_i].tolist(),
                                                                    'color_cls': match_cls,
                                                                    }

            self.global_instance = torch.cat([self.global_instance[valid_target == False], fused_instance_ids])  # 更新历史实例volume

        return contrastive_loss, valid_loss 

    def local_fusion(self, query_local, mask_local, match_iou_local, match_ins_ids_local, coords_local, color_cls_local, scale, relative_origin):
        # local没有预测出实例，保持history
        if query_local is None:
            contrastive_loss = 0
            valid_loss = False
            return contrastive_loss, valid_loss

        "only local for training loss"
        # 只与非1000实例进行损失计算
        is_valid_target_ins_ids = match_ins_ids_local != 1000
        valid_target_ins_ids = match_ins_ids_local[is_valid_target_ins_ids]
        valid_query = query_local[is_valid_target_ins_ids]

        # 计算损失
        if len(valid_query) > 1:
            # if len(torch.unique(valid_target_ins_ids)) > 1 and len(query_global) > 0:
            # if len(torch.unique(valid_target_ins_ids)) > 1:
            #     print("len(torch.unique(valid_target_ins_ids)) > 1") 
            contrastive_loss = self.compute_global_ins_fusion_loss(features=valid_query, labels=valid_target_ins_ids)
            valid_loss = True
        else:
            contrastive_loss = 0
            valid_loss = False      

        return contrastive_loss, valid_loss 
    
    def voxel_match_fast(self, A, B):
        # 将 B 转换为哈希集合，其中每个 voxel 的元组作为键
        B_set = set(map(tuple, B.tolist()))  # B 转换为 set of tuples
        
        # 检查 A 中的每个 voxel 是否在 B_set 中
        result = torch.tensor([tuple(voxel) in B_set for voxel in A.tolist()])
        
        return result

    def LLM(self, scale, instruction):
        vis_coords = self.target_tsdf_volume[scale].C - self.target_tsdf_volume[scale].C.min(dim=0)[0]  # > [0, 0, 0]
        vis_scope = vis_coords.max(dim=0)[0].cpu().tolist() 
        vis_scope = [vis_x + 10 for vis_x in vis_scope]
        vis_scope = tuple(vis_scope)

        "获取每个实例的结构化的信息"
        global_instance = deepcopy(self.global_instance)
        unique_global_instance = torch.unique(global_instance)

        ins_all = []
        for uni in unique_global_instance:
            if uni == 0:
                continue
            else:
                uni = int(uni.cpu().item())
                color_cls = self.global_ins_infos[str(uni)]['color_cls']
                color_cls_counter = Counter(color_cls)
                color_cls = color_cls_counter.most_common(1)[0][0]

                mask_3d = global_instance == uni
                mask_3d = keep_largest_connected_region_3d(vis_coords, mask_3d, vis_scope)  # 剔除离散小区域 (连通区域分析 Connected Component Analysis)
                global_instance[global_instance == uni] = mask_3d.to(global_instance.dtype) * uni

                occ = self.target_tsdf_volume[scale].C[global_instance == uni]

                # bbox = compute_3d_hbb(occ.detach().cpu().numpy())  # [x, y, z, l, w, h]
                bbox = compute_3d_obb(occ.detach().cpu().numpy())  # [x, y, z, l, w, h, theta]

                ins_all.append({'color_class': color_cls, 
                                'occ': occ, 
                                'bbox': bbox})

        use_LLM = False
        if use_LLM:
            "***************************************************************   LLM 结构化推理   ***************************************************************"
            "LLM推理: 候选实例 ➡️ 目标实例"  
            prompt = "The following are some objects and their attribute information: \n "
            for i, ins in enumerate(ins_all):
                prompt = prompt + "{ID: " + str(i) + ", " + "Object Name: " + f"{ins['color_class']}, " + "3D Center Point: " + f"{ins['bbox'][0:3]}, " + "3D Bounding Box Size: " + f"{ins['bbox'][3:6]}" + "} \n "            

            prompt = prompt + "Here is now an instruction. '" + instruction.split(" Please output segmentation masks.")[0] +  "' Find the object IDs that match the instruction based on the above information. "         
            prompt = prompt + "Think step by step, the final result is strictly in the format of <ID1, ID2, ...>"     

            with torch.no_grad():
                prompt = self.LLM_tokenizer(prompt, return_tensors='pt').to(self.LLM_model.device)
                LLM_out = self.LLM_model.generate(**prompt)
                LLM_out_text = self.LLM_tokenizer.decode(LLM_out.cpu()[0], skip_special_tokens=True)

            # 寻找输出实例ID 定义正则表达式，匹配形式 {ID1, ID2, ID3} 的子串
            pattern = r'<([^>]+)>'  # 匹配 {} 内的内容
            matches = list(re.finditer(pattern, LLM_out_text))  # 从字符串的末尾开始查找符合条件的子串, 反转字符串进行查找
            if not matches:
                matched_ids = []
            else:
                # 获取最后一个匹配的内容
                last_match = matches[-1].group(1)  # group(1) 是匹配的第一个括号内的内容
                matched_ids = [id.strip() for id in last_match.split(',') if id.strip()]  # 分割该匹配为ID并验证其格式
            
            matched_ids = [str(0)]  # debug

            LLM_global_instance = torch.zeros_like(global_instance)
            for matched_id in matched_ids:
                LLM_global_instance[global_instance == int(matched_id) + 1] = global_instance[global_instance == int(matched_id) + 1]

            # save
            save_path=f'results/global_fused_mesh_{self.scene_name[scale]}_{self.instruction_id[scale]}.ply'
            tsdf = deepcopy(self.target_tsdf_volume[scale].F)
            label2mesh(vis_coords+5, 
                        tsdf.squeeze(1), 
                        vis_scope, 
                        LLM_global_instance.unsqueeze(1), 
                        save_path)
            print("scene saved")
            "***************************************************************   LLM 结构化推理   ***************************************************************"
            
        else:
            # save
            save_path=f'results/global_fused_mesh_{self.scene_name[scale]}_{self.instruction_id[scale]}.ply'
            tsdf = deepcopy(self.target_tsdf_volume[scale].F)
            label2mesh(vis_coords+5, 
                        tsdf.squeeze(1), 
                        vis_scope, 
                        global_instance.unsqueeze(1), 
                        save_path)

            # self.save_global_mesh(scale, global_instance, vis_scope, save_path=f'results/global_fused_mesh_{self.scene_name[scale]}_{self.instruction_id[scale]}.ply')
            print("scene saved")


    # def save_global_mesh(self, scale, LLM_global_instance, vis_scope, save_path):
    #     vis_coords = self.target_tsdf_volume[scale].C - self.target_tsdf_volume[scale].C.min(dim=0)[0]
    #     vis_scope = vis_coords.max(dim=0)[0].cpu().tolist() 
    #     vis_scope = [vis_x + 10 for vis_x in vis_scope]
    #     label2mesh(vis_coords, 
    #                 self.target_tsdf_volume[scale].F.squeeze(1), 
    #                 vis_scope, 
    #                 LLM_global_instance.unsqueeze(1), 
    #                 save_path)
                    
    
    def forward_prediction_heads(self, output, mask_features):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bcl->bql", mask_embed, mask_features)

        # [B, Q, L] -> [B, Q, L] -> [B, h, Q, L] -> [B*h, Q, L]
        attn_mask = outputs_mask

        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_mask, attn_mask
    

    def reset(self, i):
        self.target_tsdf_volume[i] = PointTensor(torch.Tensor([]), torch.Tensor([]).view(0, 3).long()).cuda()
        self.global_instance = torch.empty(0).cuda()  # instance_id
        self.global_ins_infos = {}

    def convert2dense(self, coords_target_global, tsdf_target, relative_origin, scale):
        # previous frame
        global_tsdf_target = self.target_tsdf_volume[scale].F
        global_coords_target = self.target_tsdf_volume[scale].C

        # dim = (torch.Tensor(self.cfg.N_VOX).cuda() // 2 ** (self.cfg.N_LAYER - scale - 1)).int()
        dim = (torch.div(torch.Tensor(self.cfg.N_VOX).cuda(), 2 ** (self.cfg.N_LAYER - scale - 1), rounding_mode='floor')).int()
        dim_list = dim.data.cpu().numpy().tolist()

        # fuse ground truth
        if tsdf_target is not None:
            # mask voxels that are out of the FBV
            global_coords_target = global_coords_target - relative_origin
            valid_target = ((global_coords_target < dim) & (global_coords_target >= 0)).all(dim=-1)
            # combine current tsdf and global tsdf
            coords_target = torch.cat([global_coords_target[valid_target], coords_target_global])[:, :3]
            tsdf_target = torch.cat([global_tsdf_target[valid_target], tsdf_target.unsqueeze(-1)])
            # sparse to dense
            target_volume = sparse_to_dense_channel(coords_target, tsdf_target, dim_list, 1, 1, tsdf_target.device)
        else:
            target_volume = valid_target = None

        return coords_target, target_volume, valid_target

    def compute_overlap(self, points1, points2):
        # points1: [M, 3], points2: [N, 3]
        # 使用unsqueeze函数将两个点集的维度扩展为[M, 1, 3]和[1, N, 3]
        S = len(points1) + len(points2)
        points1 = points1.unsqueeze(1)
        points2 = points2.unsqueeze(0)

        # 利用广播机制计算两个点集之间的距离
        # 这里计算的是点之间的距离的平方
        distances_squared = torch.sum((points1 - points2) ** 2, dim=2)

        # 将距离为零的点对应的元素置为True，否则置为False
        same_points_mask = distances_squared == 0

        # 计算True值的数量，即相同点的数量
        I = num_same_points = same_points_mask.sum()
        U = S - I
        overlap = I / U  # IOU

        return overlap

    def save_mesh(self, scale, outputs, scene):
        if outputs is None:
            outputs = dict()
        if "scene_name" not in outputs:
            outputs['origin'] = []
            outputs['scene_tsdf'] = []
            outputs['scene_name'] = []
            outputs['scene_instance'] = []
            outputs['scene_semantic'] = []

        # only keep the newest result
        if scene in outputs['scene_name']:
            # delete old
            idx = outputs['scene_name'].index(scene)
            del outputs['origin'][idx]
            del outputs['scene_tsdf'][idx]
            del outputs['scene_name'][idx]
            del outputs['scene_instance'][idx]
            del outputs['scene_semantic'][idx]

        # scene name
        outputs['scene_name'].append(scene)

        fuse_coords = self.global_volume[scale].C
        tsdf = self.global_volume[scale].F.squeeze(-1)
        instance = self.global_instance.squeeze(-1)
        semantic = self.global_semantic.squeeze(-1)
        max_c = torch.max(fuse_coords, dim=0)[0][:3]
        min_c = torch.min(fuse_coords, dim=0)[0][:3]
        outputs['origin'].append(min_c * self.cfg.VOXEL_SIZE * (2 ** (self.cfg.N_LAYER - scale - 1)))

        ind_coords = fuse_coords - min_c
        dim_list = (max_c - min_c + 1).int().data.cpu().numpy().tolist()
        tsdf_volume = sparse_to_dense_torch(ind_coords, tsdf, dim_list, 1, tsdf.device)
        instance_volume = sparse_to_dense_torch(ind_coords, instance, dim_list, 0, instance.device)
        semantic_volume = sparse_to_dense_torch(ind_coords, semantic, dim_list, 0, semantic.device)
        outputs['scene_tsdf'].append(tsdf_volume)
        outputs['scene_instance'].append(instance_volume)
        outputs['scene_semantic'].append(semantic_volume)

        return outputs

    def forward(self, coords_target=None, tsdf_target=None, inputs=None, batch_id=None, do_ins_fusion=False, scale=2, outputs=None, save_mesh=False, ins_infos=None, qa_instruction=None):
        i = batch_id
        interval = 2 ** (self.cfg.N_LAYER - scale - 1)  # self.cfg.N_LAYER: 模型总共有3层(粗到精)；scale：当前在哪一层

        tsdf_target_all = None
        occ_target_all = None
        updated_coords_all = None
        contrastive_loss_batch = None

        scene = inputs['scene'][i]  # scene name
        instruction_id = inputs['instruction_id'][i]  # scene instruction id
        global_origin = inputs['vol_origin'][i]  # origin of global volume 全局体素初始点的坐标 [0, 0, 0]
        origin = inputs['vol_origin_partial'][i]  # origin of part volume  当前局部片段体素初始点的坐标 [fx, fy, fz]

        # each level has its corresponding voxel size
        voxel_size = self.cfg.VOXEL_SIZE * interval  # 单位: m  0.04m

        if not do_ins_fusion:  # 如果不进行实例融合
            # 获取gt
            if 'occ_list' in inputs.keys():
                # get partial gt
                occ_target = inputs['occ_list'][self.cfg.N_LAYER - scale - 1][i]
                tsdf_target = inputs['tsdf_list'][self.cfg.N_LAYER - scale - 1][i][occ_target]  # tsdf已经是由整个scene融合而来！！！
                coords_target = torch.nonzero(occ_target)
                occ_target = tsdf_target.abs() < 1  # (voxel_num, 1)

            if updated_coords_all is None:
                updated_coords_all = torch.cat([torch.ones_like(coords_target[:, :1]) * i, coords_target * interval], dim=1)
                tsdf_target_all = tsdf_target
                occ_target_all = occ_target
            else:
                coords_target = torch.cat([torch.ones_like(coords_target[:, :1]) * i, coords_target * interval], dim=1)
                updated_coords_all = torch.cat([updated_coords_all, coords_target])
                if tsdf_target_all is not None:
                    tsdf_target_all = torch.cat([tsdf_target_all, tsdf_target])
                    occ_target_all = torch.cat([occ_target_all, occ_target])

        else:
            "************************************************   FBV融合   ************************************************"
            # if this fragment is from new scene, we reinitialize backend map (三个stage都对应有一个)
            if self.scene_name[scale] is None or self.instruction_id[scale] is None or scene != self.scene_name[scale] or instruction_id != self.instruction_id[scale]:
                # if self.scene_name[scale] is not None:
                #     "一个scene-instruction处理完毕, LLM推理结构化信息, 并保存"
                #     self.LLM(scale, self.instruction)
                
                self.instruction = qa_instruction
                self.instruction_id[scale] = instruction_id
                self.scene_name[scale] = scene
                self.reset(scale)
                self.global_origin[scale] = global_origin

            # relative origin in global volume
            relative_origin = (origin - self.global_origin[scale]) / voxel_size
            relative_origin = relative_origin.cuda().long()

            "*****************************************************************   全局融合训练   *****************************************************************"
            # query_local, mask_local, match_iou_local, match_ins_ids_local, coords_all, color_cls_all, confidence_loss_batch, valid_confidence_loss = self.get_local_ins_infos(ins_info=ins_infos[i])

            # valid_target, valid_coords_target, volume_index_target = self.update_tsdf_map(coords_target[:, 1:].long(), tsdf_target, relative_origin, scale)
            # contrastive_loss_batch, valid_contrastive_loss = self.global_fusion(query_local=query_local, 
            #                                                         mask_local=mask_local, 
            #                                                         match_iou_local=match_iou_local, 
            #                                                         match_ins_ids_local=match_ins_ids_local, 
            #                                                         coords_local=coords_all, 
            #                                                         color_cls_local=color_cls_all, 
            #                                                         scale=scale, 
            #                                                         relative_origin=relative_origin,
            #                                                         valid_target=valid_target,
            #                                                         valid_coords_target=valid_coords_target,
            #                                                         volume_index_target=volume_index_target,
            #                                                         )

            "*****************************************************************   局部融合训练   *****************************************************************"
            query_local, mask_local, match_iou_local, match_ins_ids_local, coords_all, color_cls_all, confidence_loss_batch, valid_confidence_loss = self.get_local_ins_infos(ins_info=ins_infos[i])

            contrastive_loss_batch, valid_contrastive_loss = self.local_fusion(query_local=query_local, 
                                                                    mask_local=mask_local, 
                                                                    match_iou_local=match_iou_local, 
                                                                    match_ins_ids_local=match_ins_ids_local, 
                                                                    coords_local=coords_all, 
                                                                    color_cls_local=color_cls_all, 
                                                                    scale=scale, 
                                                                    relative_origin=relative_origin,
                                                                    )
            
            # save path
            # tsdf = deepcopy(self.target_tsdf_volume[scale].F)
            # vis_coords = self.target_tsdf_volume[scale].C - self.target_tsdf_volume[scale].C.min(dim=0)[0]  # > [0, 0, 0]
            # vis_scope = vis_coords.max(dim=0)[0].cpu().tolist() 
            # vis_scope = [vis_x + 10 for vis_x in vis_scope]
            # vis_scope = tuple(vis_scope)
            # global_instance = deepcopy(self.global_instance)
            # save_path=f'results/global_fused_mesh_{self.scene_name[scale]}_{self.instruction_id[scale]}.ply'
            # label2mesh(vis_coords+5, 
            #         tsdf.squeeze(1), 
            #         vis_scope, 
            #         global_instance.unsqueeze(1), 
            #         save_path)

            weight = 0.8
            loss_batch = confidence_loss_batch * weight + contrastive_loss_batch * (1 - weight)
            valid_loss = valid_confidence_loss | valid_contrastive_loss
                
                    
        if self.direct_substitude and save_mesh:
            outputs = self.save_mesh(scale, outputs, self.scene_name[scale])

        if not do_ins_fusion:
            if self.direct_substitude:
                return outputs
            else:
                return updated_coords_all, tsdf_target_all, occ_target_all
        else:
            return loss_batch, valid_loss
