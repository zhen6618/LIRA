import torch
import torch.nn as nn
import trimesh
from skimage import measure
import numpy as np
from matplotlib import cm
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from scipy.spatial import cKDTree
import plyfile
import os
from openai import OpenAI
import pickle
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

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
from datasets.visualization import label2mesh, save_coords_with_labels_to_ply
from .modules import keep_largest_connected_region_3d, evaluate_instance_segmentation
from tools.evaluation_3d import replace_rgb_in_ply

def compute_3d_hbb(points):
    "points: [N, 3]"

    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    
    center = (min_vals + max_vals) / 2
    
    l, w, h = max_vals - min_vals
    
    return np.array([np.around(center[0], decimals=0), np.around(center[1], decimals=0), np.around(center[2], decimals=0),
                     np.around(l, decimals=0), np.around(w, decimals=0), np.around(h, decimals=0)])

def compute_3d_obb(points):
    "points: [N, 3]"

    center = np.mean(points, axis=0)

    pca = PCA(n_components=3)
    pca.fit(points)
    
    rotation_matrix = pca.components_
    
    transformed_points = points - center
    transformed_points = transformed_points @ rotation_matrix.T
    
    min_vals = np.min(transformed_points, axis=0)
    max_vals = np.max(transformed_points, axis=0)
    
    l, w, h = max_vals - min_vals
    
    # Calculate the rotation Angle (rotation on the XY plane)
    theta = np.arctan2(rotation_matrix[0, 1], rotation_matrix[0, 0])

    return np.array([np.around(center[0], decimals=3), np.around(center[1], decimals=3), np.around(center[2], decimals=3), 
                    np.around(l, decimals=3), np.around(w, decimals=3), np.around(h, decimals=3), np.around(theta, decimals=3)])

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

        "t-SNE"
        self.t_SNE = []
        self.t_SNE_ids = []

        "instance fusion"
        self.global_instance = [None]  # instance_id
        self.global_ins_infos = {}

        "NN definition"
        self.num_heads = 8
        self.contrastive_hidden_dim = contrastive_hidden_dim

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

        # if self.pos_enc_type == "fourier":
        #     self.refine_pos_enc = PositionEmbeddingCoordsSine(
        #         pos_type="fourier",
        #         d_pos=refine_hidden_dim,
        #         gauss_scale=1.0,
        #         normalize=True,
        #     )
        # elif self.pos_enc_type == "sine":
        #     self.refine_pos_enc = PositionEmbeddingCoordsSine(
        #         pos_type="sine",
        #         d_pos=refine_hidden_dim,
        #         normalize=True,
        #     )
        # else:
        #     assert False, "pos enc type not known"

        self.text_transform = MLP(input_dim=4096, hidden_dim=contrastive_hidden_dim*2, output_dim=contrastive_hidden_dim, num_layers=2) 

        "*****************************************************   contrastive learning   *****************************************************"
        # self.query_feat = nn.Embedding(1, contrastive_hidden_dim)  # learnable query features
        # self.query_embed = nn.Embedding(50, hidden_dim)  # positional encoding learnable query p.e. Suppose most 50 queries
        # self.text_embed = nn.Embedding(50, hidden_dim)  

        self.box_encoder = MLP(input_dim=6, hidden_dim=contrastive_hidden_dim*2, output_dim=contrastive_hidden_dim, num_layers=3)

        self.contrastive_occ_pre_fc = nn.Linear(256, int(contrastive_hidden_dim/2))
        self.contrastive_occ_pre = Conv3d(input_dim=int(contrastive_hidden_dim/2), hidden_dim=int(contrastive_hidden_dim/2), output_dim=int(contrastive_hidden_dim/2), num_layers=12)  
        self.contrastive_occ_pre_fc_2 = nn.Linear(int(contrastive_hidden_dim/2), contrastive_hidden_dim)

        self.contrastive_num_layers = contrastive_num_layers

        self.contrastive_cross_attn_to_occ_feats = nn.ModuleList()
        self.contrastive_cross_attn_to_text_feats = nn.ModuleList()
        # self.contrastive_cross_attn_to_box_feats = nn.ModuleList()
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

            # self.contrastive_cross_attn_to_box_feats.append(
            #     CrossAttentionLayer(
            #         d_model=contrastive_hidden_dim,
            #         nhead=self.num_heads,
            #         dropout=0.0,
            #         normalize_before=False,
            #     )
            # )

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
        # self.query_embed_refine = nn.Embedding(30, refine_hidden_dim)  

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

        scene_min = coords.min(dim=0)[0]  
        scene_max = coords.max(dim=0)[0]

        scene_min = scene_min.view(-1, 3)
        scene_max = scene_max.view(-1, 3)

        pos_encodings_pcd = layer(coords[None, ...].float(), input_range=[scene_min, scene_max])  # [1, c, N]

        return pos_encodings_pcd
    
    def contrastive_iou(self, A, B):
        intersection = (A.unsqueeze(1) & B.unsqueeze(0)).sum(dim=2)  # (M, D)
        union = (A.unsqueeze(1) | B.unsqueeze(0)).sum(dim=2)  # (M, D)

        iou = intersection.float() / (union.float() + 1e-6)  # (M, D)

        return iou
    
    # def compute_contrastive_loss_all(self, features, labels, temperature=0.07):  # Refer to temperature=0.07 in CLIP
    #     """
    #     Calculate the contrast loss based on cosine similarity and support many-to-many label matching
        
    #     features: Feature vector (n, c), n represents the number of instances, and c represents the feature dimension
    #     labels: The label (n,) of each instance indicates which object each instance belongs to
    #     temperature: The temperature coefficient is used to control the similarity distribution

    #     """
    #     # Calculate the cosine similarity between features
    #     similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)  # [n_ins, n_ins]

    #     # Follow the example of EmbodedSAM and calculate each category separately
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


    def similarity_to_cluster_labels(self, similarity_matrix, threshold=0.2):
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
        Calculate the contrast loss based on cosine similarity and support many-to-many label matching
        
        features: Feature vectors (n, c), where n is the number of instances and c is the feature dimension
        labels: The label (n,) of each instance indicates which object each instance belongs to

        """
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)  # [n_ins, n_ins] [-1, 1]
        similarity_matrix = (similarity_matrix + 1) / 2  # [0, 1] 
        similarity_matrix = torch.clamp(similarity_matrix, 0, 1)

        # TODO Unbalanced handling of positive and negative sample categories

        # Calculate the target score
        target_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)

        loss = F.binary_cross_entropy(similarity_matrix, target_matrix.float())

        # cluster_labels = self.similarity_to_cluster_labels(similarity_matrix, threshold=0.5)

        return loss, similarity_matrix

    def compute_similarty_matrix(self, features):  
        """
        features: Feature vectors (n, c), where n is the number of instances and c is the feature dimension
        """
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)  # [n_ins, n_ins] [-1, 1]
        similarity_matrix = (similarity_matrix + 1) / 2  # [0, 1] 
        similarity_matrix = torch.clamp(similarity_matrix, 0, 1)

        return similarity_matrix
    
    def compute_global_ins_fusion_loss(self, features, labels, ious=None):  
        """
        Calculate the contrast loss based on cosine similarity and support many-to-many label matching
        
        features: Feature vectors (n, c), where n is the number of instances and c is the feature dimension
        labels: The label (n,) of each instance indicates which object each instance belongs to
        ious: The confidence level (n) of each instance

        """
        similarity_matrix = self.compute_similarty_matrix(features)

        target_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
        # target_ious = ious.unsqueeze(1) * ious.unsqueeze(0)
        # target_matrix = target_matrix * target_ious

        loss = F.binary_cross_entropy(similarity_matrix, target_matrix.float())

        return loss


    # def compute_contrastive_loss_two(self, t_feats, t_1_feats, t_labels, t_1_labels, temperature=0.07):  # 参考CLIP中temperature=0.07
    #         """
    #         Calculate the contrast loss based on cosine similarity
            
    #         t_feats, t_1_feats: Feature vectors (n, c), where n is the number of instances and c is the feature dimension
    #         t_labels, t_1_labels: The label (n,) of each instance indicates which object each instance belongs to
    #         temperature: The temperature coefficient is used to control the similarity distribution

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
    #             if match_ins.sum() == 1:  # When there are more than two matches, they will not be counted
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
        Calculate the mask loss
        
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

    def get_local_ins_infos(self, ins_info, relative_origin, voxel_size):
        """  img_level:
        For each image, each instance corresponds to an id:
        list 0: (num_voxels, 3), The coordinates corresponding to each voxel
        list 1: (num_instance, num_voxels), The mask corresponding to each ins is of the bool type
        list 2: 
            list 0: dict: {'id': 1, 'category': 'black chair'}
            ...
        list 3: (num_instance, text_feat_dim), Each instance corresponds to text feats
        """

        """ instance_info:
        Each batch has a list, and within each list:
        'labels': Category labels of instances (including 40 types of semantic labels)
        'masks': shape:[num_semantic_labels, num_voxels]
        'ins_ids': instance id
        """
        device = ins_info['occ_level'].device

        query_num = 0  # The number of queries = the number of ins
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

        "*****************************************************   No instance was predicted   *****************************************************"
        # if query_num == 0 or 'ins_ids' not in ins_info['gt']:
        if query_num == 0:
            confidence_loss = 0
            valid_confidence_loss = False
            return None, None, None, None, None, None, confidence_loss, valid_confidence_loss
        
        # every instance's color, class
        color_cls_all = []
        for i in range(len(ins_info['img_level'])):
            if len(ins_info['img_level'][i][2]) > 0:
                for j in range(len(ins_info['img_level'][i][2])):
                    color_cls_all.append(ins_info['img_level'][i][2][j]['category'])

        # Look for match GT instances
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
            match_gt_ins_ids_old = deepcopy(match_gt_ins_ids)
            match_gt_ins_ids[match_gt_iou < 0.6] = 1000  # base: 0.6, hard: 0.4

        else:  # If GT has no candidate instances
            match_gt_iou = torch.zeros_like(mask_all[:, 0]).to(torch.float)
            match_gt_ins_ids = torch.ones_like(mask_all[:, 0]).to(torch.long) * 1000
            match_gt_ins_ids_old = deepcopy(match_gt_ins_ids)

        "*****************************************************   contrastive learning   *****************************************************"
        # Only perform attention on non-empty Voxels to accelerate the calculation
        mask_all_no_empty = mask_all.sum(dim=0) > 0
        occ_feats_contrastive = occ_feats_all[mask_all_no_empty]  # [nV-, c]
        mask_contrastive = mask_all[:, mask_all_no_empty]  # [n_ins, nV-]  
        for no_empty_tmp in ins_info['img_level']:
            if len(no_empty_tmp[0]) > 0:
                coords_no_empty = no_empty_tmp[0][mask_all_no_empty]  # [nV-, 3]
                coords_all = no_empty_tmp[0]
                break

        "********************************   Global and local position information encoding    ********************************"
        # Obtain the coordinates of each instance
        query_contrastive = []
        pos_occ = torch.zeros((mask_contrastive.shape[1], query_num, self.contrastive_hidden_dim), device=device)  # NxBxC
        for i in range(query_num):
            mask_per_ins = mask_contrastive[i]  
            coords_local_per_ins = deepcopy(coords_no_empty[mask_per_ins])
            coords_global_per_ins = (coords_local_per_ins + relative_origin) * voxel_size  # Switch to the global coordinate system
            
            # hbox
            min_vals = coords_global_per_ins.min(dim=0).values
            max_vals = coords_global_per_ins.max(dim=0).values
            center = (min_vals + max_vals) / 2
            hbox = torch.Tensor([center[0], center[1], center[2], max_vals[0] - min_vals[0], max_vals[1] - min_vals[1], max_vals[2] - min_vals[2]])
            query_contrastive.append(self.box_encoder(hbox.unsqueeze(0).to(device)).unsqueeze(0))  # QxBxC query

            # positional encoding
            pos_occ[mask_per_ins, i] = self.get_pos_encs(coords_local_per_ins, self.contrastive_pos_enc).permute(2, 0, 1).squeeze(1)  # flatten BxCxN to NxBxC
            # pos_occ[mask_per_ins, i] = self.get_pos_encs(coords_global_per_ins, self.contrastive_pos_enc).permute(2, 0, 1).squeeze(1)  # flatten BxCxN to NxBxC
        query_contrastive = torch.cat(query_contrastive, dim=1)  # merge QxBxC query

        # # Remove global_bbox_feats
        # query_contrastive = torch.zeros((1, query_num, self.contrastive_hidden_dim)).to(text_feats.device)

        "****   occ feats   ***"
        coords_min = coords_no_empty.min(dim=0)[0] 
        coords_max = coords_no_empty.max(dim=0)[0]
        coords_no_empty = coords_no_empty - coords_min
        contrastive_spitial_shape = coords_max - coords_min
        contrastive_spitial_shape = tuple(contrastive_spitial_shape.tolist())

        # Convert occ_feats in advance preprocessing
        occ_feats_contrastive = self.contrastive_occ_pre_fc(occ_feats_contrastive)
        occ_feats_contrastive = self.contrastive_occ_pre(x=occ_feats_contrastive,
                                                        coords=coords_no_empty,  # [nV-, 3]
                                                        spitial_shape=contrastive_spitial_shape,
                                                        )
        occ_feats_contrastive = self.contrastive_occ_pre_fc_2(occ_feats_contrastive)

        # NxBxC
        occ_feats_contrastive = occ_feats_contrastive.unsqueeze(1).repeat(1, query_num, 1)  # [nV-, B, c]  

        "****   attn mask   ***"
        # [Q, L] -> [B, Q, L] -> [B, h, Q, L] -> [B*h, Q, L]
        # mask_contrastive = mask_contrastive.unsqueeze(0)  # [1, n_ins, nV-]=[B, Q, L]
        # mask_contrastive = mask_contrastive.unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1)
        mask_contrastive = ~mask_contrastive  # False: 允许attention; True: 不允许attention
        mask_contrastive = mask_contrastive.detach()

        # text mask
        text_mask = torch.ones(query_num, query_num, dtype=torch.bool, device=device)
        text_mask[torch.arange(query_num), torch.arange(query_num)] = False
        # [Q, L] -> [B, Q, L] -> [B, h, Q, L] -> [B*h, Q, L]
        text_mask = text_mask.unsqueeze(0).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1)
        text_mask = text_mask.detach()
        # NxBxC
        text_feats_contrastive = text_feats.unsqueeze(0)  # [n_ins, 1, c]
        # pos_text = self.text_embed.weight[:query_num, :].unsqueeze(1)  # positional encoding

        "********************************   transformers    ********************************"
        # box_feats_contrastive = query_contrastive.clone()  
        for j in range(self.contrastive_num_layers):  # transformer loop
            'cross-attention to occ_feats'
            # query_contrastive = self.contrastive_cross_attn_to_occ_feats[j](
            #     query_contrastive, occ_feats_contrastive,
            #     memory_mask=mask_contrastive,  # For a binary mask, a True value indicates that the corresponding position is not allowed to attend
            #     memory_key_padding_mask=None,  # here we do not apply masking on padded region
            #     pos=pos_occ,
            # )
            # Here, the query of QxBxC is placed from position Q to position B
            query_contrastive = self.contrastive_cross_attn_to_occ_feats[j](
                query_contrastive, occ_feats_contrastive,
                memory_mask=None,   # here we do not apply masking on attn region
                memory_key_padding_mask=mask_contrastive,  # For a binary mask, a True value indicates that the corresponding position is not allowed to attend
                pos=pos_occ,
            )

            # for qj in range(query_num):
            #     query_contrastive_tmp = query_contrastive[qj].unsqueeze(0)
            #     occ_feats_contrastive_tmp = occ_feats_contrastive[mask_contrastive_copy[qj]]
            #     pos_occ_tmp = pos_occ[qj]

            #     query_contrastive_tmp = self.contrastive_cross_attn_to_occ_feats[j](
            #         query_contrastive_tmp, occ_feats_contrastive_tmp,
            #         # memory_mask=mask_contrastive[:, qj, mask_contrastive_copy[qj]].unsqueeze(1),  # For a binary mask, a True value indicates that the corresponding position is not allowed to attend
            #         memory_key_padding_mask=None,  # here we do not apply masking on padded region
            #         pos=pos_occ_tmp,
            #     )   
            #     query_contrastive[qj] = query_contrastive_tmp.squeeze(0)

            'cross-attention to text_feats'
            query_contrastive = self.contrastive_cross_attn_to_text_feats[j](
                query_contrastive, text_feats_contrastive,
                # memory_mask=text_mask,  # For a binary mask, a True value indicates that the corresponding position is not allowed to attend
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
            )

            'cross-attention to box_feats'
            # query_contrastive = self.contrastive_cross_attn_to_box_feats[j](
            #     query_contrastive, box_feats_contrastive,
            #     # memory_mask=text_mask,  # For a binary mask, a True value indicates that the corresponding position is not allowed to attend
            #     memory_key_padding_mask=None,  # here we do not apply masking on padded region
            # )

            'FFN'
            query_contrastive = self.contrastive_ffn_layers[j](
                query_contrastive
            )

        # MLP is converted to the contrast loss space and the confidence space
        query_contrastive_transform = self.contrastive_transform(query_contrastive.squeeze(0))  # [n_ins, c]  
        query_confidence_transform = self.confidence_transform(query_contrastive.squeeze(0))  # [n_ins, 1]

        "confidence loss"
        query_confidence_transform = query_confidence_transform.squeeze(1).sigmoid()  
        query_confidence_transform = torch.clamp(query_confidence_transform, 0, 1)

        match_gt_iou = torch.clamp(match_gt_iou, 0, 1)

        # if sum(match_gt_ins_ids == 1000) > 0:
        #     print("confidence 1000")
        confidence_loss = F.mse_loss(query_confidence_transform, match_gt_iou)
        valid_confidence_loss = True

        '***********************************************************************   if testing start   ***********************************************************************'
        # match_gt_ins_ids_old[query_confidence_transform < 0.6] = 1000  # base: 0.6, hard: 0.4
        # match_gt_ins_ids = deepcopy(match_gt_ins_ids_old)
        '***********************************************************************   if testing end   ***********************************************************************'

        "invalid=1000 drop"
        valid_fusion = match_gt_ins_ids != 1000
        if sum(valid_fusion) > 0:
            query_contrastive_transform = query_contrastive_transform[valid_fusion]
            mask_all = mask_all[valid_fusion]
            match_gt_iou = match_gt_iou[valid_fusion]
            match_gt_ins_ids = match_gt_ins_ids[valid_fusion]
            color_cls_all = [a for a, b in zip(color_cls_all, valid_fusion) if b]
        else:
            query_contrastive_transform, mask_all, match_gt_iou, match_gt_ins_ids, coords_all, color_cls_all = None, None, None, None, None, None

        # self.t_SNE.append(query_contrastive_transform.detach().cpu().numpy())
        # self.t_SNE_ids.append(match_gt_ins_ids.detach().cpu().numpy())
        # # visualize_tsne_features(self.t_SNE)

        return query_contrastive_transform, mask_all, match_gt_iou, match_gt_ins_ids, coords_all, color_cls_all, confidence_loss, valid_confidence_loss

    def compute_iou_overlap(self, instances, is_overlap=True):
        instances_expanded = instances.unsqueeze(0)  # shape: (1, N, M)
        instances_expanded_T = instances.unsqueeze(1)  # shape: (N, 1, M)
        
        intersection = torch.sum(instances_expanded & instances_expanded_T, dim=2)  # shape: (N, N)
        
        union = torch.sum(instances_expanded | instances_expanded_T, dim=2)  # shape: (N, N)
        
        iou_matrix = intersection.float() / union.float()
        
        iou_matrix.fill_diagonal_(1.0)

        if is_overlap:
            return intersection
        else:
            return iou_matrix


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
                    'query': [c]  
                    'ious': [1, ]  # confidence
                    'match_count': [1, ]  # Cumulative matching times
                    'target_ins_ids': list  # target ins_id
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
        fused_instance_ids = torch.zeros(volume_index_target.sum(), dtype=self.global_instance.dtype, device=device)  # Initialize the instance label [nV', ]

        # Unify the occ within the FBV
        dim = (torch.Tensor(self.cfg.N_VOX).cuda() // 2 ** (self.cfg.N_LAYER - scale - 1)).int()
        dim_list = dim.data.cpu().numpy().tolist()

        ins_global = self.global_instance[valid_target]
        ins_volume_global = sparse_to_dense_channel(valid_coords_target, ins_global.unsqueeze(-1), dim_list, 1, 0, device)  # [96, 96, 96, 1]
        ins_global = ins_volume_global[volume_index_target].squeeze(1)  # [nV']

        # local did not predict an instance, so keep the history
        if query_local is None:
            fused_instance_ids = ins_global
            self.global_instance = torch.cat([self.global_instance[valid_target == False], fused_instance_ids])  # Update the historical instance volume
            contrastive_loss = 0
            valid_loss = False
            return contrastive_loss, valid_loss

        mask_local = mask_local.transpose(1, 0)  # [nV, N] 
        ins_volume_local = sparse_to_dense_channel(coords_local, mask_local, dim_list, mask_local.shape[1], 0, device)  # [96, 96, 96, N]
        mask_local = ins_volume_local[volume_index_target].transpose(1, 0)  # [N, nV']

        # # Look for the part where the historical voxel overlaps with the current voxel
        # coords_global = self.target_tsdf_volume[scale].C  
        # coords_global = coords_global - relative_origin  # Switch to the local coordinate system
        # valid_global = ((coords_global < dim) & (coords_global >= 0)).all(dim=-1)

        # coords_tmp_global = coords_global[valid_global] 
        # valid_tmp_global = self.voxel_match_fast(coords_tmp_global, coords_local)
        # valid_global[valid_global.clone()] = valid_tmp_global.to(torch.bool).to(device)

        # Historical instance information
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
            if uni_ins_global == 0:  # background instance id=0
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

        # Indicate which locations are global (note that global is always placed ahead of local)
        is_global = torch.zeros_like(target_ins_ids).to(torch.bool)
        is_global[:len(target_ins_ids_global)] = True

        'local and global fusion for training loss'
        # # Loss calculation is only performed with non-1000 instances
        # is_valid_target_ins_ids = target_ins_ids != 1000
        # valid_target_ins_ids = target_ins_ids[is_valid_target_ins_ids]
        # valid_query = query[is_valid_target_ins_ids]

        "only local for training loss"
        # Loss calculation is only performed with non-1000 instances
        is_valid_target_ins_ids = target_ins_ids != 1000
        valid_target_ins_ids = target_ins_ids[is_valid_target_ins_ids]
        valid_query = query[is_valid_target_ins_ids]

        # loss
        if len(valid_query) > 1:
            # if len(torch.unique(valid_target_ins_ids)) > 1:
            #     print("len(torch.unique(valid_target_ins_ids)) > 1") 
            contrastive_loss = self.compute_global_ins_fusion_loss(features=valid_query, labels=valid_target_ins_ids)
            valid_loss = True
        else:
            contrastive_loss = 0
            valid_loss = False      
        
        # Instance matching and fusion
        similarity_matrix = self.compute_similarty_matrix(query)

        # # overlap > thresd matching
        # similarity_auxilary = self.compute_iou_overlap(masks, is_overlap=True)
        # similarity_auxilary = similarity_auxilary > 50
        # similarity_matrix = similarity_matrix * similarity_auxilary.float()

        # # Only the same text category matches # Adding this restriction actually makes the effect worse
        # n = len(color_clses)
        # class_matrix = torch.zeros_like(similarity_matrix)  # Initialize the (n, n) matrix
        # for i in range(n):
        #     for j in range(n):
        #         if color_clses[i] == color_clses[j]:
        #             class_matrix[i, j] = 1
        # similarity_matrix = similarity_matrix * class_matrix

        match_ids = self.similarity_to_cluster_labels(similarity_matrix, threshold=0.2)  # After converting to a distance matrix, the smaller value should be taken

        with torch.no_grad():
            unique_match_ids, unique_counts = torch.unique(match_ids, return_counts=True)
            _, unique_sorted_indices = torch.sort(unique_counts, descending=True)  # Sort in descending order of occurrence frequency and prioritize the fusion of those with more matching times. This helps to eliminate the mask area of invalid instances
            unique_match_ids = unique_match_ids[unique_sorted_indices]

            for match_id in unique_match_ids:
                match_i = match_ids == match_id
                match_global = match_i & is_global

                cur_mask = masks[match_i].any(dim=0)  # [nV, ]
                match_cls = [a for a, b in zip(color_clses, match_i) if b]

                if sum(match_global) > 0:  # Inherit the historical instance numbers
                    global_idx = torch.where(match_global == True)[0][0]
                    assign_ins_id = unique_ins_global[1:][global_idx].item()

                    # update instance id
                    fused_instance_ids[(fused_instance_ids.clone() == 0) & cur_mask] = assign_ins_id  
                    # update query
                    self.global_ins_infos[str(assign_ins_id)]['query'] = self.global_ins_infos[str(assign_ins_id)]['query'] * self.global_ins_infos[str(assign_ins_id)]['match_count'] + query[match_i][1:].sum(0)
                    self.global_ins_infos[str(assign_ins_id)]['query'] = self.global_ins_infos[str(assign_ins_id)]['query'] / (self.global_ins_infos[str(assign_ins_id)]['match_count'] + match_i.sum() - 1)
                    # update ious
                    self.global_ins_infos[str(assign_ins_id)]['ious'] = self.global_ins_infos[str(assign_ins_id)]['ious'] * self.global_ins_infos[str(assign_ins_id)]['match_count'] + ious[match_i][1:].sum(0)
                    self.global_ins_infos[str(assign_ins_id)]['ious'] = self.global_ins_infos[str(assign_ins_id)]['ious'] / (self.global_ins_infos[str(assign_ins_id)]['match_count'] + match_i.sum() - 1)
                    # update match_count
                    self.global_ins_infos[str(assign_ins_id)]['match_count'] = self.global_ins_infos[str(assign_ins_id)]['match_count'] + match_i.sum() - 1
                    # update target ins ids   
                    self.global_ins_infos[str(assign_ins_id)]['target_ins_ids'] = self.global_ins_infos[str(assign_ins_id)]['target_ins_ids'] + target_ins_ids[match_i][1:].tolist()   
                    # update color class
                    self.global_ins_infos[str(assign_ins_id)]['color_cls'] = self.global_ins_infos[str(assign_ins_id)]['color_cls'] + match_cls[1:]   

                else:  # Create a new instance number

                    # update instance id
                    fused_instance_ids[(fused_instance_ids.clone() == 0) & cur_mask] = max_ins_id_global + 1
                    max_ins_id_global += 1
                    
                    self.global_ins_infos[str(max_ins_id_global)] = {'query': query[match_i].sum(0) / match_i.sum(),
                                                                    'ious': ious[match_i].sum(0) / match_i.sum(),
                                                                    'match_count': match_i.sum(),
                                                                    'target_ins_ids': target_ins_ids[match_i].tolist(),
                                                                    'color_cls': match_cls,
                                                                    }

            self.global_instance = torch.cat([self.global_instance[valid_target == False], fused_instance_ids])  # Update the historical instance volume

        return contrastive_loss, valid_loss 

    def local_fusion(self, query_local, mask_local, match_iou_local, match_ins_ids_local, coords_local, color_cls_local, scale, relative_origin):
        # local did not predict an instance, so keep the history
        if query_local is None:
            contrastive_loss = 0
            valid_loss = False
            return contrastive_loss, valid_loss

        "only local for training loss"
        # Loss calculation is only performed with non-1000 instances
        is_valid_target_ins_ids = match_ins_ids_local != 1000
        valid_target_ins_ids = match_ins_ids_local[is_valid_target_ins_ids]
        valid_query = query_local[is_valid_target_ins_ids]

        # loss
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
        # Convert B to a hash set, where each voxel tuple serves as the key
        B_set = set(map(tuple, B.tolist()))  # B transforms to set of tuples
        
        # Check whether each voxel in A is in B_set
        result = torch.tensor([tuple(voxel) in B_set for voxel in A.tolist()])
        
        return result

    def LLM(self, scale, instruction, voxel_size):
        vis_coords = self.target_tsdf_volume[scale].C - self.target_tsdf_volume[scale].C.min(dim=0)[0]  # > [0, 0, 0]
        vis_scope = vis_coords.max(dim=0)[0].cpu().tolist() 
        vis_scope = [vis_x + 10 for vis_x in vis_scope]
        vis_scope = tuple(vis_scope)

        "Eliminate discrete small regions (Connected Component Analysis)"
        global_instance = deepcopy(self.global_instance)
        unique_global_instance = torch.unique(global_instance)

        for uni in unique_global_instance:
            if uni == 0:
                continue
            else:
                uni = int(uni.cpu().item())

                # cur_global_ins_infos = self.global_ins_infos[str(uni)]
                # # Matching number limit
                # if cur_global_ins_infos['match_count'] < 3: 
                #     global_instance[global_instance == uni] = 0

                # mask quantity limit
                mask_3d = global_instance == uni
                mask_3d = keep_largest_connected_region_3d(vis_coords, mask_3d, vis_scope)  
                if mask_3d.sum() < 100:
                    global_instance[global_instance == uni] = 0
                else:
                    global_instance[global_instance == uni] = mask_3d.to(global_instance.dtype) * uni

        "***************************************************************   Calculation of segmentation accuracy for candidate instances start   ***************************************************************"
        gt_ply_path = "/root/paddlejob/workspace/env_run/zhouzhen05/Data/zhouzhen05/Data/ScanNetV2_ply/"
        qa_path = "/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/grounding_scene_qa_infos_hard/"
        mapping_save_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/grounding_scene_instance_infos_mapping/'
        gt_ins_infos_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/grounding_scene_instance_infos/'
        volume_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data/zhouzhen05/Data/all_tsdf_9_1/'
        scan_results_path = 'scan_results_base_new/'
        if not os.path.exists(scan_results_path):
            os.mkdir(scan_results_path)

        '*********************   pred instance (Turn off the data augmentation of scans in advance!!!)   *********************'
        pred_coords = (self.target_tsdf_volume[scale].C * voxel_size).cpu().numpy()  # (nV, 3)
        pred_instance = global_instance.cpu().numpy().astype(np.int64)  # (nV,)

        # KDTree Nearest neighbor search
        pred_kdtree = cKDTree(pred_coords)

        gt_path = gt_ply_path + f'{self.scene_name[scale]}_vh_clean_2.ply'
        gt_plydata = plyfile.PlyData.read(gt_path)
        gt_coords = np.vstack([gt_plydata['vertex'][dim] for dim in ('x', 'y', 'z')]).T

        # pred instance
        _, pred_indices = pred_kdtree.query(gt_coords)
        pred_mapped_instance = pred_instance[pred_indices]
        # save_coords_with_labels_to_ply(gt_coords, pred_mapped_instance, "results/pred_mapped_ins.ply")

        # Statistical prediction instance information
        pred_ins_all = []
        unique_pred_mapped_instance = np.unique(pred_mapped_instance)
        for uni in unique_pred_mapped_instance:
            if uni == 0:
                continue
            else:
                color_cls = self.global_ins_infos[str(uni)]['color_cls']
                color_cls_counter = Counter(color_cls)
                color_cls = color_cls_counter.most_common(1)[0][0]

                mask_3d = pred_mapped_instance == uni
                if mask_3d.sum() < 100:
                    continue
                else:
                    occ = gt_coords[mask_3d]
                    # bbox = compute_3d_hbb(gt_coords)  # [x, y, z, l, w, h]
                    bbox = compute_3d_obb(occ)  # [x, y, z, l, w, h, theta]

                    # pred_ins_all.append({'color_class': color_cls, 'mask': mask_3d, 'occ': occ, 'bbox': bbox})
                    pred_ins_all.append({'color_class': color_cls, 'mask': mask_3d, 'bbox': bbox})

        '*********************   gt instance   *********************'
        gt_ins_path = volume_path + f'{self.scene_name[scale]}/full_instance_layer_interpolate0.npz'
        gt_instance_volume = np.load(gt_ins_path, allow_pickle=True)
        gt_instance_volume = gt_instance_volume.f.arr_0
        gt_tsdf_path = volume_path + f'{self.scene_name[scale]}/full_tsdf_layer0.npz'
        gt_tsdf_volume = np.load(gt_tsdf_path, allow_pickle=True)
        gt_tsdf_volume = gt_tsdf_volume.f.arr_0
        gt_tsdf_volume = gt_tsdf_volume.reshape(-1)

        gt_shape = gt_instance_volume.shape
        gt_indices = np.indices((gt_shape[0], gt_shape[1], gt_shape[2]))
        gt_coords_volume = np.stack(gt_indices, axis=-1).reshape(-1, 3)
        gt_instance_volume = gt_instance_volume.reshape(-1)

        gt_valid_tsdf = np.abs(gt_tsdf_volume) < 1
        gt_coords_volume = gt_coords_volume[gt_valid_tsdf]
        gt_instance_volume = gt_instance_volume[gt_valid_tsdf]
        gt_coords_volume = gt_coords_volume * voxel_size + self.old_origin.cpu().numpy()

        gt_kdtree = cKDTree(gt_coords_volume)
        _, gt_indices = gt_kdtree.query(gt_coords)
        gt_mapped_instance = gt_instance_volume[gt_indices]
        # save_coords_with_labels_to_ply(gt_coords, gt_mapped_instance, "results/gt_mapped_ins.ply")

        # Obtain candidate instances
        qa_path = qa_path + self.scene_name[scale] + "_" + str(self.instruction_id[scale].tolist()) + "_qa.pkl"
        with open(qa_path, 'rb') as file:
            scene_qa = pickle.load(file)
            qa = scene_qa[0]
            qa_origin = deepcopy(qa)
        with open(mapping_save_path + self.scene_name[scale] + '.pkl', 'rb') as file:
            ins_mapping_dict = pickle.load(file)

            for i in range(len(qa['instance ids'])):
                qa['instance ids'][i] = ins_mapping_dict[str(qa['instance ids'][i])]

            for j in range(len(qa['candidate'])): 
                qa['candidate'][j] = ins_mapping_dict[str(qa['candidate'][j])]   
        
        qa_candidate_ids = np.array(qa['candidate'])
        gt_mapped_instance[~np.isin(gt_mapped_instance, qa_candidate_ids)] = 0
        # save_coords_with_labels_to_ply(gt_coords, gt_mapped_instance, "results/gt_mapped_ins.ply")

        with open(gt_ins_infos_path + self.scene_name[scale] + '_instance.pkl', 'rb') as file: 
            gt_ins_infos = pickle.load(file)

        gt_ins_all = []
        for candidate_id in qa_origin['candidate']:
            color_cls = gt_ins_infos[candidate_id]['color'] + " " + gt_ins_infos[candidate_id]['semantic label']
            mask_3d = gt_mapped_instance == ins_mapping_dict[str(candidate_id)]
            gt_ins_all.append({'color_class': color_cls.lower(), 'mask': mask_3d})

        # ap25, ap50, ap75, mAP = evaluate_instance_segmentation([pred_ins_all], [gt_ins_all])
        # print(f"ap25: {round(ap25, 3)}, ap50: {round(ap50, 3)}, mAP: {round(mAP, 3)}")

        gt_ins_all_final = []
        for candidate_id in qa_origin['instance ids']:
            color_cls = gt_ins_infos[candidate_id]['color'] + " " + gt_ins_infos[candidate_id]['semantic label']
            mask_3d = gt_mapped_instance == ins_mapping_dict[str(candidate_id)]
            gt_ins_all_final.append({'color_class': color_cls.lower(), 'mask': mask_3d})

        "*** visualization ply ***"
        # replace_rgb_in_ply(gt_path, pred_mapped_instance, f'results/{self.scene_name[scale]}_colored.ply')

        "*** save per-scan result ***"
        scan_results = {'candidate_pred_ins': pred_ins_all, 'candidate_gt_ins': gt_ins_all, 'instruction': instruction, 'final_gt_ins': gt_ins_all_final, 'gt_coords': gt_coords}
        with open(scan_results_path + self.scene_name[scale] + "_" + str(self.instruction_id[scale].tolist()) + "_scan_results.pkl", 'wb') as f:
            pickle.dump(scan_results, f)

        "***************************************************************   Calculation of segmentation accuracy for candidate instances end   ***************************************************************"

        use_LLM = False
        if use_LLM:
            "***************************************************************   LLM Structured reasoning   ***************************************************************"
            "LLM inference: Candidate instance ➡️ target instance"
            prompt = "The following are some objects and their attribute information: \n "
            for i, ins in enumerate(pred_ins_all):
                prompt = prompt + "{ID: " + str(i) + ", " + "Color and Class: " + f"{ins['color_class']}, " + "Position Coordinate (x, y, z) (meter): " + f"{ins['bbox'][0:3]}, " + "Size (meter^3): " + str(np.round(np.prod(ins['bbox'][3:6]), 3)) + "} \n "        
            if len(pred_ins_all) == 0:
                prompt = prompt + "There is no object. \n "

            prompt = prompt + "Here is now an instruction. '" + instruction.split(" Please give all objects that are helpful in inferring final targets")[0] +  "' \n Find objects and their IDs that match the instruction based on the above object information. If it involves calculating the distance between objects, please make the judgment based on the Euclidean distance between the position coordinates of the objects. "         
            prompt = prompt + "When it comes to comparisons between objects, if there is only one object, no comparison is needed. The final result is strictly in the format of <0, 1, 2, ...>, strictly end with '<>' symbol. '<>' contains object IDs, and the ID inside should be in numeric form. If there is no object in the final result, return <>. "   
            prompt = prompt + "Let's think step by step."
            # prompt = prompt + " Finally, write a brief sentence to summarize."

            use_gpt = True
            if use_gpt:
                client = OpenAI()
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",  # gpt-4o, gpt-4o-mini
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ]
                )
                LLM_out_text = completion.choices[0].message.content

            else:
                with torch.no_grad():
                    prompt = self.LLM_tokenizer(prompt, return_tensors='pt').to(self.LLM_model.device)
                    LLM_out = self.LLM_model.generate(**prompt)
                    LLM_out_text = self.LLM_tokenizer.decode(LLM_out.cpu()[0], skip_special_tokens=True)

            # Find the output instance ID definition regular expression that matches the substring of the form {ID1, ID2, ID3}
            pattern = r'<([^>]+)>'  # Match the content within {}
            matches = list(re.finditer(pattern, LLM_out_text))  # Start looking for the substring that meets the conditions from the end of the string, and reverse the string for the search
            if not matches:
                matched_ids = []
            else:
                # Get the last matching content
                last_match = matches[-1].group(1)  # group(1) is the content within the first matching parenthesis
                matched_ids = [id.strip() for id in last_match.split(',') if id.strip()]  # Split this match into an ID and verify its format
            
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
            "***************************************************************   LLM Structured reasoning   ***************************************************************"
            
        else:
            # save
            save_path=f'results/global_fused_mesh_{self.scene_name[scale]}_{self.instruction_id[scale]}.ply'
            # tsdf = deepcopy(self.target_tsdf_volume[scale].F)
            # label2mesh(vis_coords+5, 
            #             tsdf.squeeze(1), 
            #             vis_scope, 
            #             global_instance.unsqueeze(1), 
            #             save_path)

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

    def reset(self, i):
        self.target_tsdf_volume[i] = PointTensor(torch.Tensor([]), torch.Tensor([]).view(0, 3).long()).cuda()
        self.global_instance = torch.empty(0).cuda()  # instance_id
        self.global_ins_infos = {}

    def forward(self, coords_target=None, tsdf_target=None, inputs=None, batch_id=None, do_ins_fusion=False, scale=2, outputs=None, save_mesh=False, ins_infos=None, qa_instruction=None):
        i = batch_id
        interval = 2 ** (self.cfg.N_LAYER - scale - 1)  # self.cfg.N_LAYER: The model has a total of 3 layers (from coarse to fine); scale: At which level is it currently

        tsdf_target_all = None
        occ_target_all = None
        updated_coords_all = None
        contrastive_loss_batch = None

        scene = inputs['scene'][i]  # scene name
        instruction_id = inputs['instruction_id'][i]  # scene instruction id
        global_origin = inputs['vol_origin'][i]  # origin of global volume The coordinates of the initial point of the global voxel [0, 0, 0]
        origin = inputs['vol_origin_partial'][i]  # origin of part volume  The coordinates of the initial point of the current local fragment voxel [fx, fy, fz]
        old_origin = inputs['old_origin'][i]  # full volume The origin in the world coordinate system

        # each level has its corresponding voxel size
        voxel_size = self.cfg.VOXEL_SIZE * interval  # 单位: m  0.04m

        if not do_ins_fusion:  # If instance fusion is not carried out
            # obtain gt
            if 'occ_list' in inputs.keys():
                # get partial gt
                occ_target = inputs['occ_list'][self.cfg.N_LAYER - scale - 1][i]
                tsdf_target = inputs['tsdf_list'][self.cfg.N_LAYER - scale - 1][i][occ_target]  # tsdf has already been integrated from the entire scene！！！
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
            "************************************************   FBV fusion   ************************************************"
            # if this fragment is from new scene, we reinitialize backend map (Each of the three stages corresponds to one)
            if self.scene_name[scale] is None or self.instruction_id[scale] is None or scene != self.scene_name[scale] or instruction_id != self.instruction_id[scale]:
                # if self.scene_name[scale] is not None:
                #     "Once a scene-instruction is processed, the LLM inferences the structured information and saves it"
                #     # MDS_analysis(self.t_SNE, self.t_SNE_ids)
                #     self.LLM(scale, self.instruction, voxel_size)
                
                self.old_origin = old_origin
                self.instruction = qa_instruction
                self.instruction_id[scale] = instruction_id
                self.scene_name[scale] = scene
                self.reset(scale)
                self.global_origin[scale] = global_origin

            # relative origin in global volume
            relative_origin = (origin - self.global_origin[scale]) / voxel_size
            relative_origin = relative_origin.cuda().long()

            "*****************************************************************   Global fusion training and inference   *****************************************************************"
            query_local, mask_local, match_iou_local, match_ins_ids_local, coords_all, color_cls_all, confidence_loss_batch, valid_confidence_loss = self.get_local_ins_infos(ins_info=ins_infos[i], 
                                                                                                                                                                              relative_origin=relative_origin,
                                                                                                                                                                              voxel_size=voxel_size,
                                                                                                                                                                              )

            valid_target, valid_coords_target, volume_index_target = self.update_tsdf_map(coords_target[:, 1:].long(), tsdf_target, relative_origin, scale)
            contrastive_loss_batch, valid_contrastive_loss = self.global_fusion(query_local=query_local, 
                                                                    mask_local=mask_local, 
                                                                    match_iou_local=match_iou_local, 
                                                                    match_ins_ids_local=match_ins_ids_local, 
                                                                    coords_local=coords_all, 
                                                                    color_cls_local=color_cls_all, 
                                                                    scale=scale, 
                                                                    relative_origin=relative_origin,
                                                                    valid_target=valid_target,
                                                                    valid_coords_target=valid_coords_target,
                                                                    volume_index_target=volume_index_target,
                                                                    )

            "*****************************************************************   Local fusion training   *****************************************************************"
            # query_local, mask_local, match_iou_local, match_ins_ids_local, coords_all, color_cls_all, confidence_loss_batch, valid_confidence_loss = self.get_local_ins_infos(ins_info=ins_infos[i])

            # contrastive_loss_batch, valid_contrastive_loss = self.local_fusion(query_local=query_local, 
            #                                                         mask_local=mask_local, 
            #                                                         match_iou_local=match_iou_local, 
            #                                                         match_ins_ids_local=match_ins_ids_local, 
            #                                                         coords_local=coords_all, 
            #                                                         color_cls_local=color_cls_all, 
            #                                                         scale=scale, 
            #                                                         relative_origin=relative_origin,
            #                                                         )
            
            # # save path
            # tsdf = deepcopy(self.target_tsdf_volume[scale].F)
            # vis_coords = self.target_tsdf_volume[scale].C - self.target_tsdf_volume[scale].C.min(dim=0)[0]  # > [0, 0, 0]
            # vis_scope = vis_coords.max(dim=0)[0].cpu().tolist() 
            # vis_scope = [vis_x + 10 for vis_x in vis_scope]
            # vis_scope = tuple(vis_scope)

            # # global_instance = deepcopy(self.global_instance)
            # "Eliminate discrete small regions (Connected Component Analysis)"
            # global_instance = deepcopy(self.global_instance)
            # unique_global_instance = torch.unique(global_instance)
            # for uni in unique_global_instance:
            #     if uni == 0:
            #         continue
            #     else:
            #         uni = int(uni.cpu().item())
            #         mask_3d = global_instance == uni
            #         mask_3d = keep_largest_connected_region_3d(vis_coords, mask_3d, vis_scope)  
            #         global_instance[global_instance == uni] = mask_3d.to(global_instance.dtype) * uni

            # save_path=f'results/debug_global_fused_mesh_{self.scene_name[scale]}_{self.instruction_id[scale]}.ply'  # /root/paddlejob/workspace/env_run/zhouzhen05/Data/zhouzhen05/Data/scans/scene0009_01
            # label2mesh(vis_coords+5, 
            #         tsdf.squeeze(1), 
            #         vis_scope, 
            #         global_instance.unsqueeze(1), 
            #         save_path)


            weight = 0.5
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

def visualize_tsne_features(features, save_path='tsne_visualization.png', random_seed=42):
    """
    The given feature data is dimensionally reduced and visualized by using t-SNE, and finally saved as an image.
    
    参数：
        features: numpy (200, 128)，Represents 200 samples, with 128-dimensional features for each sample
        save_path: 
        random_seed: 
    """

    features = np.concatenate(features, axis=0)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / norms 
    print('features: ', features.shape)

    # Reduce 128-dimensional features to 2 dimensions using t-SNE
    if len(features) > 30:
        perplexity = 30
    else:
        perplexity = len(features) - 1

    tsne = TSNE(n_components=2, random_state=random_seed, perplexity=perplexity)
    features_2d = tsne.fit_transform(features)
    print('features_2d: ', features_2d.shape)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c='blue', s=30)
    # plt.xlabel("Dimension 1")
    # plt.ylabel("Dimension 2")
    plt.title("t-SNE Visualization")
    plt.grid(False)
    
    plt.savefig(save_path, dpi=300)
    plt.show()


def MDS_analysis(features, labels):
    features = np.concatenate(features, axis=0)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    similarity_matrix = np.dot(features, features.T)
    similarity_matrix = (similarity_matrix + 1) / 2
    similarity_matrix = np.clip(similarity_matrix, 0, 1)

    labels = np.concatenate(labels, axis=0)

    distance_matrix = 1 - similarity_matrix

    # mds = MDS(n_components=2, dissimilarity='precomputed')
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    embedding = mds.fit_transform(distance_matrix)

    # Standardize the embedding to make it more compact
    scaler = StandardScaler()
    embedding = scaler.fit_transform(embedding) * 0.5  

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    cmap = plt.get_cmap('tab10')  
    colors = [cmap(i) for i in labels_encoded] 

    plt.figure(figsize=(8, 6))

    for i in range(len(embedding)):
        plt.scatter(
            embedding[i, 0], embedding[i, 1], 
            s=100, linewidths=2, 
            edgecolors=colors[i], facecolors='none' 
        )

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    handles, labels_unique = [], []
    for i, lbl in enumerate(np.unique(labels)):
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor=cmap(i), markersize=8, linewidth=2))
        labels_unique.append(f'{i+1}')
    # plt.legend(handles, labels_unique, title="Instance Labels")
    plt.legend(handles, labels_unique)

    # plt.figure(figsize=(8, 6))
    # scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels_encoded, cmap='viridis', s=100)

    # plt.title("2D Visualization of Instances")
    # plt.xlabel("Dimension 1")
    # plt.ylabel("Dimension 2")

    # plt.colorbar(scatter, label='Instance Label')
    # plt.colorbar(scatter)

    plt.savefig("MDS.png", dpi=300)

    # plt.show()
