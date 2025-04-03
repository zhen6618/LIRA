import torch
import torch.nn as nn
from torchsparse.tensor import PointTensor
from neu_utils import sparse_to_dense_channel, sparse_to_dense_torch
from .modules import ConvGRU
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import squareform
import torch.nn.functional as F
from .mask3dformer import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP, Conv3d
from .voxel_position_encoding import PositionEmbeddingCoordsSine


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
        self.global_origin = [None, None, None]
        self.target_tsdf_volume = [None, None, None]

        self.global_instance = [None]  # instance_id
        self.global_semantic = [None]  # semantic_id

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

        self.contrastive_transform = MLP(input_dim=contrastive_hidden_dim, hidden_dim=contrastive_hidden_dim*2, output_dim=int(contrastive_hidden_dim/2), num_layers=2)

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
    #         if ins_gt_id == -100:
    #             continue
            
    #         label_mask = (labels == ins_gt_id)

    #         label_matrix = label_mask.unsqueeze(1).repeat(1, len(labels)) & label_mask.unsqueeze(0).repeat(len(labels), 1)  # [n_ins, n_ins]
    #         similarity_sum_id = torch.sum(torch.exp(similarity_matrix[label_matrix] / temperature))

    #         label_matrix_all = label_mask.unsqueeze(1).repeat(1, len(labels)) | label_mask.unsqueeze(0).repeat(len(labels), 1)  # [n_ins, n_ins]
    #         similarity_sum_all = torch.sum(torch.exp(similarity_matrix[label_matrix_all] / temperature))

    #         contrastive_loss.append(-torch.log(similarity_sum_id / similarity_sum_all))  # 0 ~ 1
        
    #     if len(contrastive_loss) == 0:
    #         valid_loss = False
    #         print("all -100")
    #         return features.sum() * 0, valid_loss
    #     else: 
    #         contrastive_loss = sum(contrastive_loss) / len(contrastive_loss)  # loss per instance
    #         valid_loss = True

    #         return contrastive_loss, valid_loss


    def similarity_to_cluster_labels(self, similarity_matrix, threshold=0.5):
        # Ensure similarity_matrix is numpy array
        similarity_matrix = similarity_matrix.detach().cpu().numpy() 

        # Convert similarity to distance (1 - similarity)
        distance_matrix = 1 - similarity_matrix

        # Use DBSCAN to find clusters
        clustering = DBSCAN(eps=threshold, min_samples=1, metric='precomputed')
        cluster_labels = clustering.fit_predict(distance_matrix)

        return cluster_labels

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

    def local_ins_fusion(self, ins_info, loss_type):
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
        volume_shape = (96, 96, 96)
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

        # image_ids
        image_ids = []
        for i in range(len(ins_info['img_level'])):
            if len(ins_info['img_level'][i][4]) > 0:
                image_ids.append(ins_info['img_level'][i][4])
        if len(image_ids) > 0:
            image_ids = torch.cat(image_ids, dim=0)  # [n_ins,]

        "*****************************************************   没有预测出实例   *****************************************************"
        if query_num == 0:

            refined_masks = None
            contrastive_loss = refine_loss = 0
            valid_loss = False
            similarity_matrix = None

            return refined_masks, contrastive_loss, refine_loss, valid_loss, similarity_matrix
        
        # 1. 寻找match GT实例 
        if len(ins_info['gt']['ins_ids']) > 0:
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
            match_gt_ins_ids[match_gt_iou < 0.1] = -100  # 认为是错误类 
        else:  # 无GT实例
            match_gt_ins_ids = torch.ones(query_num, dtype=torch.long, device=device) * (-100)

        "*****************************************************   contrastive learning start   *****************************************************"
        if loss_type == 'contrastive' or loss_type == 'all':
            # QxBxC query
            query_contrastive = self.query_feat.weight.repeat(query_num, 1).unsqueeze(1) 
            # query_contrastive_embed = self.query_embed.weight[:query_num, :].unsqueeze(1)  # 位置编码

            # 只对非空的voxel进行attention
            mask_all_no_empty = mask_all.sum(dim=0) > 0
            occ_feats_contrastive = occ_feats_all[mask_all_no_empty]  # [nV-, c]
            mask_contrastive = mask_all[:, mask_all_no_empty]  # [n_ins, nV-]  
            for no_empty_tmp in ins_info['img_level']:
                if len(no_empty_tmp[0]) > 0:
                    coords_no_empty = no_empty_tmp[0][mask_all_no_empty]  # [nV-, 3]
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

            # MLP转换到对比损失空间
            query_contrastive_transform = self.contrastive_transform(query_contrastive.squeeze(1))  # [n_ins, c]

            # ---  compute contrastive loss  ---
            # 去除-100的ins_id，减少contrastive loss损失干扰
            valid_ins_ids = match_gt_ins_ids != -100
            if len(valid_ins_ids) < 0:
                valid_loss = False
                contrastive_loss = 0
                similarity_matrix = None
                print("all -100")

            else:
                valid_image_ids = image_ids[valid_ins_ids]
                valid_query_contrastive_transform = query_contrastive_transform[valid_ins_ids]
                valid_match_gt_ins_ids = match_gt_ins_ids[valid_ins_ids]

                if len(torch.unique(valid_match_gt_ins_ids)) > 1:  # 286
                    print("exist negative sample: ", valid_match_gt_ins_ids)

                "***************************   每两帧之间计算对比损失   ***************************"
                # contrastive_loss = []
                # contrastive_loss_valid = []
                # for contrastive_i in range(8):
                #     t_valid_image_ids = valid_image_ids == contrastive_i  # t
                #     t_1_valid_image_ids = valid_image_ids == contrastive_i + 1  # t+1

                #     t_valid_query_contrastive_transform = valid_query_contrastive_transform[t_valid_image_ids]
                #     t_1_valid_query_contrastive_transform = valid_query_contrastive_transform[t_1_valid_image_ids]

                #     t_valid_match_gt_ins_ids = valid_match_gt_ins_ids[t_valid_image_ids]
                #     t_1_valid_match_gt_ins_ids = valid_match_gt_ins_ids[t_1_valid_image_ids]

                #     contrastive_loss_i, valid_loss_i = self.compute_contrastive_loss_two(t_valid_query_contrastive_transform, 
                #                                                                  t_1_valid_query_contrastive_transform,
                #                                                                  t_valid_match_gt_ins_ids,
                #                                                                  t_1_valid_match_gt_ins_ids
                #                                                                  )
                #     contrastive_loss.append(contrastive_loss_i)
                #     contrastive_loss_valid.append(valid_loss_i)
                
                # if sum(contrastive_loss_valid) == 0:
                #     valid_loss = False
                #     contrastive_loss = 0
                #     # print("all no match")
                #     # print("valid_image_ids: ", valid_image_ids)
                #     # print("contrastive_loss: ", contrastive_loss)
                #     # print("contrastive_loss_valid: ", contrastive_loss_valid)
                # else:
                #     valid_loss = True
                #     contrastive_loss = sum(contrastive_loss) / sum(contrastive_loss_valid)

                "***************************   9帧一起算对比损失   ***************************"
                if len(valid_match_gt_ins_ids) <= 1:
                    contrastive_loss = 0
                    valid_loss = False
                    similarity_matrix = None
                else:
                    contrastive_loss, similarity_matrix = self.compute_contrastive_loss_all_ce(valid_query_contrastive_transform, valid_match_gt_ins_ids)
                    valid_loss = True

        else:
            contrastive_loss = 0

        "*****************************************************   contrastive learning end   *****************************************************"


        "*****************************************************   3D Occ Mask Refinement start   *****************************************************"
        if loss_type == 'refine' or loss_type == 'all':
            # QxBxC query
            # query_refine = self.refine_transform(query_contrastive)  # [n_ins, 1, c]  
    
            # ---------   instance/feat/mask fusion  ---------
            query_refine_num = len(torch.unique(match_gt_ins_ids))

            refine_mask_all = []
            refine_ins_ids = []
            for ins_gt_id in torch.unique(match_gt_ins_ids):
                refine_ins_ids.append(ins_gt_id)
                is_same_ins = match_gt_ins_ids == ins_gt_id
                refine_mask_all.append(mask_all[is_same_ins].any(dim=0))  # 如果ins_id = -100，也全部Union

            refine_ins_ids = torch.stack(refine_ins_ids, dim=0)  # [n_ins_refine, ]
            refine_mask_all = torch.stack(refine_mask_all, dim=0)  # [n_ins_refine, nV]

            # QxBxC query
            query_refine = self.query_feat_refine.weight.repeat(query_refine_num, 1).unsqueeze(1) 
            query_refine_embed = self.query_embed_refine.weight[:query_refine_num, :].unsqueeze(1)  # 位置编码

            # ---------   mask features  ---------
            coords_refine = ins_info['img_level'][0][0]

            # coords最小归于0
            refine_coords_min = coords_refine.min(dim=0)[0] 
            refine_coords_max = coords_refine.max(dim=0)[0]
            coords_refine = coords_refine - refine_coords_min
            refine_spitial_shape = refine_coords_max - refine_coords_min
            refine_spitial_shape = tuple(refine_spitial_shape.tolist())

            occ_feats_refine = self.refine_occ_pre_fc(occ_feats_all)
            mask_features_refine = self.mask_feat_refine(x=occ_feats_refine,  # [nV, c]
                                                        coords=coords_refine,  # [nV, 3]
                                                        spitial_shape=refine_spitial_shape,
                                                        )
            mask_features_refine=mask_features_refine.unsqueeze(0).permute(0, 2, 1)  # BxCxL
            
            occ_feats_refine = self.refine_occ_pre(x=occ_feats_refine,  # [nV, c]
                                                    coords=coords_refine,  # [nV, 3]
                                                    spitial_shape=refine_spitial_shape,
                                                    )
            occ_feats_refine = occ_feats_refine.unsqueeze(1)  # [nV, 1, c]

            # ---------   positional encoding  --------- 
            pos_occ_refine = self.get_pos_encs(coords_refine, self.refine_pos_enc)
            # flatten BxCxN to NxBxC
            pos_occ_refine = pos_occ_refine.permute(2, 0, 1)

            # ---------   transformer loop  ---------        
            predictions_mask = []

            # 第一次使用先验mask [Q, L] -> [B, Q, L] -> [B, h, Q, L] -> [B*h, Q, L]
            attn_mask = refine_mask_all.unsqueeze(0)  # [1, n_ins, nV]=[B, Q, L]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            attn_mask = ~attn_mask  # False: 允许attention; True: 不允许attention
            attn_mask = attn_mask.detach()  # TODO 检查所有梯度的反向传播信息应该有还是没有

            for j in range(self.refine_num_layers):  # transformer loop
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False  

                # attention: cross-attention first
                query_refine = self.refine_cross_attention_layers[j](
                    query_refine, occ_feats_refine,
                    memory_mask=attn_mask,  # True means not to participate
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos_occ_refine, query_pos=query_refine_embed,
                )

                query_refine = self.refine_self_attention_layers[j](
                    query_refine, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_refine_embed,
                )

                # FFN
                query_refine = self.refine_ffn_layers[j](
                    query_refine
                )

                outputs_mask, attn_mask = self.forward_prediction_heads(query_refine, mask_features_refine)
                predictions_mask.append(outputs_mask.squeeze(0))

            refined_masks = predictions_mask[-1].sigmoid() >= 0.5  # [Q, L] # TODO 如果mask的voxel数量小于一定的阈值，认为时干扰项，剔除！

            # ---  compute loss  ---
            # 1. 寻找GT实例 
            gt_masks_refine = []
            for ins_id_refine in refine_ins_ids:
                if len(ins_info['gt']['ins_ids']) == 0:  # 无GT实例
                    gt_masks_refine.append(torch.zeros(len(ins_info['img_level'][0][0]), dtype=torch.float, device=device).unsqueeze(0))
                else:
                    idx_refine = torch.where(ins_info['gt']['ins_ids'] == ins_id_refine)[0]
                    if len(idx_refine) == 0:
                        gt_masks_refine.append(torch.zeros_like(ins_info['gt']['masks'][0]).unsqueeze(0))
                    else:
                        gt_masks_refine.append(ins_info['gt']['masks'][idx_refine])
            gt_masks_refine = torch.cat(gt_masks_refine, dim=0)  # [n_ins_refine, nV]

            # 2. 计算mask损失
            refine_loss = self.compute_refine_loss(predictions_mask, gt_masks_refine.float())  # [Q, L] 
            valid_loss = True    

        else:
            refined_masks = None
            refine_loss = 0

        "*****************************************************   3D Occ Mask Refinement end   *****************************************************"

        return refined_masks, contrastive_loss, refine_loss, valid_loss, similarity_matrix
    
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
        self.global_semantic = torch.empty(0).cuda()  # semantic_id

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

    def global_ins_fusion(self, scale, global_valid, relative_origin, ins_info, current_coords):

        overlap_threshold = 0.05

        current_coords = current_coords + relative_origin  # 转换到全局坐标系
        current_voxel_id_all = ins_info[0]  # 每个voxel的id  current_coords_in_updates: 只对tsdf < 1的进行融合

        global_instance = self.global_instance[global_valid]  # [N, ] instance_id
        global_semantic = self.global_semantic[global_valid]  # [N, ] semantic_id
        if len(self.global_instance) > 0:
            max_global_instance_id = torch.max(self.global_instance)
        else:
            max_global_instance_id = 0

        new_current_instance = torch.zeros_like(current_voxel_id_all)  # [N, ] 0表示空语义
        new_current_semantic = torch.zeros_like(current_voxel_id_all)  # [N, ]

        # 先保持历史实例，新的预测实例直接在此基础上进行覆盖
        history_matrix = (self.target_tsdf_volume[scale].C[global_valid].unsqueeze(1) == current_coords.unsqueeze(0)).all(dim=-1)
        if history_matrix.shape[0] != 0:
            matched_current_indices = history_matrix.any(dim=0)  # 找到global中每个点对应current中的匹配索引
            matched_global_indices = history_matrix.float().argmax(dim=0)  # 获取global中的匹配索引

            new_current_instance[matched_current_indices] = global_instance[matched_global_indices[matched_current_indices]].reshape(-1).to(new_current_instance.dtype)
            new_current_semantic[matched_current_indices] = global_semantic[matched_global_indices[matched_current_indices]].reshape(-1).to(new_current_semantic.dtype)

        # current更新
        num_current_instance = len(ins_info[1])
        increment_count = 1  # 用于新增instance_id数
        for i in range(num_current_instance):  # 对于每一个instance
            cls = ins_info[1][i]['category_id']

            if cls in global_semantic:
                cls_global_index = global_semantic == cls  # in current volume
                ins_global = global_instance[cls_global_index]
                ins_global_all = torch.unique(ins_global)

                match_flag = False
                # 取出global中所有是这个instance_id的点
                for ins_id in ins_global_all:
                    comparison_global_index = self.global_instance == ins_id  # in global volume
                    comparison_global_coords = self.target_tsdf_volume[scale].C[comparison_global_index.view(-1)]

                    overlap = self.compute_overlap(comparison_global_coords, current_coords[current_voxel_id_all == i+1])
                    if overlap > overlap_threshold:
                        new_current_instance[current_voxel_id_all == i+1] = int(ins_id)  # 更新 和global_instance_id融合
                        new_current_semantic[current_voxel_id_all == i+1] = cls
                        match_flag = True
                        break

                """
                如果历史预测不对，可以在此处加上融合网络！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
                """

                if match_flag == False:
                    # current_instance未与global匹配上
                    new_current_instance[current_voxel_id_all == i+1] = int(max_global_instance_id + increment_count)  # 更新 新增一个instance_id
                    new_current_semantic[current_voxel_id_all == i+1] = cls
                    increment_count = increment_count + 1

            # current_instance未与global匹配上
            else:
                new_current_instance[current_voxel_id_all == i+1] = int(max_global_instance_id + increment_count)  # 更新 新增一个instance_id
                new_current_semantic[current_voxel_id_all == i+1] = cls
                increment_count = increment_count + 1

        return new_current_instance, new_current_semantic
    

    def update_map(self, target_volume, valid_target, relative_origin, scale, new_current_instance=None, new_current_semantic=None):
        '''
        Replace Hidden state/tsdf in global Hidden state/tsdf volume by direct substitute corresponding voxels
        :param value: (Tensor) fused feature (N, C)
        :param coords: (Tensor) updated coords (N, 3)
        :param target_volume: (Tensor) tsdf volume (DIM_X, DIM_Y, DIM_Z, 1)
        :param valid: (Tensor) mask: 1 represent in current FBV (N,)
        :param valid_target: (Tensor) gt mask: 1 represent in current FBV (N,)
        :param relative_origin: (Tensor), origin in global volume, (3,)
        :param scale:
        :return:
        '''
        self.global_instance = torch.cat([self.global_instance[valid_target == False], new_current_instance])  # 直接替换历史实例
        self.global_semantic = torch.cat([self.global_semantic[valid_target == False], new_current_semantic])

        # target
        if target_volume is not None:
            target_volume = target_volume.squeeze()
            self.target_tsdf_volume[scale].F = torch.cat([self.target_tsdf_volume[scale].F[valid_target == False], target_volume[target_volume.abs() < 1].unsqueeze(-1)])
            target_coords = torch.nonzero(target_volume.abs() < 1) + relative_origin

            self.target_tsdf_volume[scale].C = torch.cat([self.target_tsdf_volume[scale].C[valid_target == False], target_coords])

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

    def forward(self, coords=None, tsdf_target=None, inputs=None, batch_id=None, do_ins_fusion=False, scale=2, outputs=None, save_mesh=False, ins_infos=None, loss_type=None):
        i = batch_id
        interval = 2 ** (self.cfg.N_LAYER - scale - 1)  # self.cfg.N_LAYER: 模型总共有3层(粗到精)；scale：当前在哪一层

        tsdf_target_all = None
        occ_target_all = None
        updated_coords_all = None
        refined_mask = None
        contrastive_loss_batch = None
        refine_loss_batch = None

        scene = inputs['scene'][i]  # scene name
        global_origin = inputs['vol_origin'][i]  # origin of global volume 全局体素初始点的坐标 [0, 0, 0]
        origin = inputs['vol_origin_partial'][i]  # origin of part volume  当前局部片段体素初始点的坐标 [fx, fy, fz]

        if scene != self.scene_name[scale] and self.scene_name[scale] is not None and self.direct_substitude:
            outputs = self.save_mesh(scale, outputs, self.scene_name[scale])

        # if this fragment is from new scene, we reinitialize backend map (三个stage都对应有一个)
        if self.scene_name[scale] is None or scene != self.scene_name[scale]:
            self.scene_name[scale] = scene
            self.reset(scale)
            self.global_origin[scale] = global_origin

        # each level has its corresponding voxel size
        voxel_size = self.cfg.VOXEL_SIZE * interval  # 单位: m  0.04m

        # relative origin in global volume
        relative_origin = (origin - self.global_origin[scale]) / voxel_size
        relative_origin = relative_origin.cuda().long()

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
            batch_ind = torch.nonzero(coords[:, 0] == i).squeeze(1)
            # coords_b = coords[batch_ind, 1:].long() // interval
            coords_b = torch.div(coords[batch_ind, 1:].long(), interval, rounding_mode='floor')  # [N_voxels, 3]
            # tsdf_target_b = tsdf_target[batch_ind]

            do_global_fusion = False

            if do_global_fusion:
                "********************   与全局进行实例融合   ********************"
                # convert to dense: 1. convert sparse feature to dense feature; 2. combine current feature coordinates and
                # previous feature coordinates within FBV from our backend map to get new feature coordinates (updated_coords)
                updated_coords, target_volume, valid_target = self.convert2dense(
                    coords_b,
                    tsdf_target,
                    relative_origin,
                    scale)

                # get fused gt
                tsdf_target_b = target_volume[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]]  # (voxel_num, 1)
                # occ_target_b = tsdf_target_b.abs() < 1  # (voxel_num, 1)

            else:
                "********************   只进行当前FBV内的实例融合   ********************"
                updated_coords = coords_b
                # occ_target_b = tsdf_target_b.abs() < 1  # (voxel_num, 1)

            dim = (torch.div(torch.Tensor(self.cfg.N_VOX).cuda(), 2 ** (self.cfg.N_LAYER - scale - 1), rounding_mode='floor')).int()
            dim_list = dim.data.cpu().numpy().tolist()

            "************************************************   9张图片的实例整体融合一次   ************************************************"
            if do_global_fusion:
                for img_i in range(len(ins_infos[i])):
                    current_instance_volume = sparse_to_dense_channel(coords_b, ins_infos[i][img_i][0].unsqueeze(1), dim_list, c=1, default_val=0, device=tsdf_target.device)
                    ins_infos[i][img_i][0] = current_instance_volume[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]].squeeze(1)

                    # new_current_instance: [N, ], new_current_semantic: [N, ]
                    new_current_instance, new_current_semantic = self.global_ins_fusion(scale=scale, 
                                                                                        global_valid=valid_target,
                                                                                        relative_origin=relative_origin,
                                                                                        ins_info=ins_infos[i][img_i],
                                                                                        current_coords=updated_coords)

                    # feed back to global volume (direct substitute)  fragment融合， global直接替代
                    self.update_map(target_volume, valid_target, relative_origin, scale, new_current_instance.view(-1, 1), new_current_semantic.view(-1, 1))

            else:
                # refined_masks: [n_ins, nV]
                refined_mask, contrastive_loss_batch, refine_loss_batch, valid_loss, similarity_matrix = self.local_ins_fusion(ins_info=ins_infos[i], loss_type=loss_type)            
                    
        if self.direct_substitude and save_mesh:
            outputs = self.save_mesh(scale, outputs, self.scene_name[scale])

        if not do_ins_fusion:
            if self.direct_substitude:
                return outputs
            else:
                return updated_coords_all, tsdf_target_all, occ_target_all
        else:
            return refined_mask, contrastive_loss_batch, refine_loss_batch, valid_loss, similarity_matrix
