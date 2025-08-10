import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsparse.tensor import PointTensor
from loguru import logger
import re
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from matplotlib import cm
import torchvision.transforms as transforms
from PIL import Image
import os
import time
import open3d as o3d
from plyfile import PlyData, PlyElement

from models.modules import SPVCNN
from neu_utils import apply_log_transform
from .fusion import Fusion
from ops.back_project import back_project, ins_back_project, mask_reshape, rs_ins_back_project
from ops.generate_grids import generate_grid
from skimage import measure
import trimesh
from datasets.visualization import visualize_mesh
# from models.grounding_2d_llava_grounding import Grounding_2D
from models.reason_seg import Reason_Seg

class NeuConNet(nn.Module):
    '''
    Coarse-to-fine network.
    '''

    def __init__(self, cfg):
        super(NeuConNet, self).__init__()
        self.cfg = cfg
        self.n_scales = len(cfg.THRESHOLDS) - 1

        channels = [96, 48, 24]  

        if self.cfg.FUSION.FUSION_ON:
            # GRU Fusion
            self.fusion = Fusion(cfg, 
                                channels, 
                                contrastive_hidden_dim=128, 
                                refine_hidden_dim=256, 
                                contrastive_num_layers=3,
                                refine_num_layers=3,
                                )
        # grounding_2d
        # self.grounding_2d_model = Grounding_2D()
        self.reason_seg_model = Reason_Seg()
        # print("self.reason_seg_model device: ", self.reason_seg_model.model.device)


    def get_target(self, coords, inputs, scale):
        '''
        Won't be used when 'fusion_on' flag is turned on
        :param coords: (Tensor), coordinates of voxels, (N, 4) (4 : Batch ind, x, y, z)
        :param inputs: (List), inputs['tsdf_list' / 'occ_list']: ground truth volume list, [(B, DIM_X, DIM_Y, DIM_Z)]
        :param scale:
        :return: tsdf_target: (Tensor), tsdf ground truth for each predicted voxels, (N,)
        :return: occ_target: (Tensor), occupancy ground truth for each predicted voxels, (N,)
        '''
        with torch.no_grad():
            tsdf_target = inputs['tsdf_list'][scale]
            occ_target = inputs['occ_list'][scale]
            coords_down = coords.detach().clone().long()
            # 2 ** scale == interval
            coords_down[:, 1:] = (coords[:, 1:] // 2 ** scale)
            tsdf_target = tsdf_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]
            occ_target = occ_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]
            return tsdf_target, occ_target

    def get_xyzrgb_targets(self, coords, inputs, scale):
        with torch.no_grad():
            rgb_target = inputs['rgb_list'][scale]

            coords_down = coords.detach().clone().long()
            # 2 ** scale == interval
            coords_down[:, 1:] = (coords[:, 1:] // 2 ** scale)
            rgb_target = rgb_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3], :]
            xyzrgb_target = torch.concat([coords[:, 1:], rgb_target], dim=-1)

            return xyzrgb_target

    def get_panoptic_targets(self, coords, inputs, scale, batch):
        valid_semantic_labels = [3, 4, 5, 6, 7, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36]
        class_name_dict = {3: 'cabinet', 4: 'bed', 5: 'chair', 
                        6: 'sofa', 7: 'table', 10: 'bookshelf', 
                        11: 'picture', 12: 'counter', 14: 'desk', 16: 'curtain', 24: 'refrigerator',
                        28: 'shower curtain', 33: 'toilet', 34: 'sink', 36: 'bathtub'}

        with torch.no_grad():
            semantic_target = inputs['semantic_list'][scale]
            instance_target = inputs['instance_list'][scale]

            coords_down = coords.detach().clone().long()
            # 2 ** scale == interval
            coords_down[:, 1:] = (coords[:, 1:] // 2 ** scale)
            semantic_label = semantic_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]  # [num_voxel, ] class_id
            instance_label = instance_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]  # [num_voxel, ] instance_id
 
            panoptic_targets = {}

            labels = []
            masks = []
            ins_ids_all = []
            unique_ids = torch.unique(instance_label)
            for unique_id in unique_ids:
                unique_indice = torch.where(instance_label == unique_id)[0]

                cur_label = torch.argmax(torch.bincount(semantic_label[unique_indice].to(torch.int)))
                if cur_label not in valid_semantic_labels:
                    continue

                labels.append(cur_label) # In case instance labels and category labels are not equal, select the most
                masks.append((instance_label == unique_id).unsqueeze(0))
                ins_ids_all.append(unique_id)

            if len(labels) != 0:
                panoptic_targets['labels'] = torch.tensor(labels).to(instance_target.device)  # (num_instance, )
                panoptic_targets['masks'] = torch.cat(masks, dim=0)  # (num_instance, num_voxel)
                panoptic_targets['ins_ids'] = torch.tensor(ins_ids_all).to(instance_target.device).to(labels[0].dtype)  # (num_instance, )

                del labels, masks

            return panoptic_targets

    def upsample(self, pre_feat, pre_coords, interval, num=8):
        '''

        :param pre_feat: (Tensor), features from last level, (N, C)
        :param pre_coords: (Tensor), coordinates from last level, (N, 4) (4 : Batch ind, x, y, z)
        :param interval: interval of voxels, interval = scale ** 2
        :param num: 1 -> 8
        :return: up_feat : (Tensor), upsampled features, (N*8, C)
        :return: up_coords: (N*8, 4), upsampled coordinates, (4 : Batch ind, x, y, z)
        '''
        with torch.no_grad():
            pos_list = [1, 2, 3, [1, 2], [1, 3], [2, 3], [1, 2, 3]]
            n, c = pre_feat.shape
            up_feat = pre_feat.unsqueeze(1).expand(-1, num, -1).contiguous()
            up_coords = pre_coords.unsqueeze(1).repeat(1, num, 1).contiguous()
            for i in range(num - 1):
                up_coords[:, i + 1, pos_list[i]] += interval

            up_feat = up_feat.view(-1, c)
            up_coords = up_coords.view(-1, 4)

        return up_feat, up_coords

    def forward(self, inputs, outputs):
        bs = inputs['proj_matrices'].shape[0]
        device = inputs['proj_matrices'].device

        # ----generate new coords----
        # coords = generate_grid(self.cfg.N_VOX, interval=1)[0]
        # up_coords = []
        # for b in range(bs):
        #     up_coords.append(torch.cat([torch.ones(1, coords.shape[-1]).to(coords.device) * b, coords]))
        # up_coords = torch.cat(up_coords, dim=1).permute(1, 0).contiguous()  # [96*96*96, 4(bxyz)]

        KRcam = inputs['proj_matrices'][:, :, 0].permute(1, 0, 2, 3).contiguous()

        loss_dict = {}

        panoptic_targets = []
        ins_pred_batch_level = []
        contrastive_losses = []
        refine_losses = []
        refined_masks = []
        valid_losses = [] 
        similarity_matrices = []

        for batch in range(bs):
            "***********************************************   geometry reconstruction   ************************************************"
            # ----   qa   ---- 
            qa_instruction = inputs['qa']['instruction'][batch]
            qa_candidates = [candidate[batch] for candidate in inputs['qa']['candidate']]
            qa_candidates = torch.stack(qa_candidates)

            up_coords, tsdf_target, occ_target = self.fusion(inputs=inputs, batch_id=batch)  
            # 可视化panoptic target
            # for batch in range(bs):
            #     batch_ind = torch.nonzero(up_coords[:, 0] == batch).squeeze(1)
            #     generate_mesh(up_coords[batch_ind][:, 1:], panoptic_targets[batch], volume_shape, 'results/panoptic_target.ply', mode='panoptic')

            # ----   generate panoptic target   ---- 
            panoptic_targets.append(self.get_panoptic_targets(up_coords, inputs, 0, batch))

            ins_pred_batch_level.append({})

            # ----   If there is no GT instance in the current FBV, skip fusion and refinement directly   ---- 
            if len(panoptic_targets[batch]) == 0:
                valid_losses.append(False)
                continue
            isin_gt = torch.isin(panoptic_targets[batch]['ins_ids'], qa_candidates)
            if not isin_gt.any():
                valid_losses.append(False)
                continue

            "***********************************************   reason_seg start   ************************************************"
            batch_ind = torch.nonzero(up_coords[:, 0] == batch).squeeze(1)
            up_coords_batch = up_coords[batch_ind]
            imgs = inputs['imgs'][batch]  # [N_views, 3, H, W] = [9, 3, 480, 640]

            reason_seg_imgs = imgs.detach().cpu() 
            reason_seg_imgs = reason_seg_imgs / 255  # 0 ~ 1

            "reason seg 多张图片并行推理 [9, 3, 480, 640]"
            selected_img_ids = [0, 3, 6]

            rs_imgs_np_all = []
            for rs_i in range(reason_seg_imgs.size(0)):
                if not rs_i in selected_img_ids:
                    continue

                rs_imgs_np = reason_seg_imgs[rs_i].permute(1, 2, 0).numpy() 
                rs_imgs_np = (rs_imgs_np * 255).astype(np.uint8)
                ori_height, ori_width, _ = rs_imgs_np.shape  
                resized_height, resized_width = int(ori_height / 4), int(ori_width / 4) 
                rs_imgs_np = cv2.resize(rs_imgs_np, (1024, 1024))
                rs_imgs_np_all.append(rs_imgs_np)

            with torch.no_grad():
                prompt = qa_instruction
                # t_rs_start = time.time()
                # masks: [N_mask, 1024, 1024], rs_text_feats: [N_mask, 4096], rs_img_feats: [N_mask, 256, 1024, 1024]
                response_all, masks_all, rs_text_feats_all, rs_img_feats_all = self.reason_seg_model.inference_parallel(prompt, rs_imgs_np_all) 
                # print("rs inference time: ", time.time() - t_rs_start, "s.")

                rs_img_feats_all = rs_img_feats_all.to(torch.float32) 

            " Collect each image "
            ins_pred_img_level = []
            occ_feats_volume_batch = []  
            for rs_i in range(len(selected_img_ids)):  # reason_seg_imgs: [9, 3, 480, 640]

                masks = masks_all[rs_i].float().unsqueeze(1)
                masks = F.interpolate(masks, size=(resized_height, resized_width), mode='nearest') 
                masks = masks.squeeze(1).bool() 
                # save_mask(masks[0], "mask_bw.png") 

                rs_img_feats = F.interpolate(rs_img_feats_all[rs_i].unsqueeze(0), size=(resized_height, resized_width), mode='bilinear', align_corners=False)

                # back-project to voxel
                ins_text = [match.strip() for match in re.findall(r'([\w\s]+)\s\[SEG\]', response_all[rs_i])]
                ins_volume = []  

                depth = inputs['depth'][batch][selected_img_ids[rs_i]].unsqueeze(0).unsqueeze(0)
                depth = F.interpolate(depth, size=(resized_height, resized_width), mode='nearest')
                depth = depth.squeeze(0).squeeze(0)
                
                for m in range(len(masks)):
                    ins_volume.append(rs_ins_back_project(up_coords_batch[:, 1:], inputs['vol_origin_partial'][batch], self.cfg.VOXEL_SIZE,  
                                                        masks[m].unsqueeze(0), KRcam[:, batch], selected_img_ids[rs_i], depth)) 
                if len(ins_text) != len(ins_volume):  
                    continue
                occ_feats_volume_batch.append(rs_ins_back_project(up_coords_batch[:, 1:], inputs['vol_origin_partial'][batch], self.cfg.VOXEL_SIZE, 
                                                                rs_img_feats.squeeze(0), KRcam[:, batch], selected_img_ids[rs_i], depth))  

                # vis
                # if len(ins_volume) > 0:
                #     label2mesh(up_coords_batch[:, 1:], tsdf_target[batch_ind], tuple(self.cfg.N_VOX), ins_volume[0].float().unsqueeze(1), f'results/label_target_{selected_img_ids[rs_i]}.ply')

                """
                对于每张图像, 将每个实例对应一个id:
                list 0: (num_voxels, 3), 每个voxel对应的坐标
                list 1: (num_instance, num_voxels), 每个ins对应的mask, bool类型
                list 2: 
                    list 0: dict: {'id': 1, 'category': 'black chair', 'sem_id': sem class}
                    ...
                list 3: (num_instance, text_feat_dim), 每个实例对应的text feats
                list 4: (num_instance,) image index
                """
                ins_preds = []
                ins_id_count = 1
                ins_preds.append([])  # list 0
                ins_preds.append([])  # list 1
                ins_preds.append([])  # list 2
                ins_preds.append([])  # list 3
                ins_preds.append([])  # list 4

                for m in range(len(ins_volume)):
                    if ins_volume[m].sum() >= 50:  
                        ins_preds[1].append(ins_volume[m])
                        ins_preds[2].append({'id': ins_id_count, 'category': ins_text[m]})
                        ins_preds[3].append(rs_text_feats_all[rs_i][m].to(torch.float32))
                        ins_preds[4].append(rs_i)
                        ins_id_count += 1

                if len(ins_preds[1]) > 0:
                    ins_preds[0] = up_coords_batch[:, 1:]
                    ins_preds[1] = torch.stack(ins_preds[1])
                    ins_preds[3] = torch.stack(ins_preds[3])
                    ins_preds[4] = torch.tensor(ins_preds[4]).to(ins_preds[3].device)

                ins_pred_img_level.append(ins_preds)

            occ_feats_volume_batch = torch.stack(occ_feats_volume_batch)  # [9, N, 256]
            occ_mask_batch = torch.all(occ_feats_volume_batch == 0, dim=2)  # [9, N]
            occ_mask_batch = ~occ_mask_batch
            occ_mask_batch = occ_mask_batch.sum(dim=0) 
            occ_mask_batch[occ_mask_batch == 0] = 1  

            occ_feats_volume_b = occ_feats_volume_batch.sum(dim=0)  # [N, 256]
            occ_feats_volume_b = occ_feats_volume_b / occ_mask_batch.unsqueeze(1)  

            ins_pred_batch_level[batch]['img_level'] = ins_pred_img_level
            ins_pred_batch_level[batch]['occ_level'] = occ_feats_volume_b

            del occ_mask_batch, occ_feats_volume_b

            # targets
            occ_mask_batch = torch.all(occ_feats_volume_batch == 0, dim=2)
            occ_mask_batch = ~occ_mask_batch
            contrastive_iou_mask = []

            for k in range(len(occ_mask_batch)):
                contrastive_iou_mask.append(occ_mask_batch[k].unsqueeze(0).repeat(len(ins_pred_img_level[k][2]), 1))
            contrastive_iou_mask = torch.cat(contrastive_iou_mask, dim=0)  # [n_SEG, nV]

            panoptic_targets[batch]['contrastive_iou_mask'] = contrastive_iou_mask
            ins_pred_batch_level[batch]['gt'] = panoptic_targets[batch]

            # 3D instance fusion
            fusion_mode = 'contrastive'  # ['contrastive', 'refine', 'all']
            refined_mask, contrastive_loss_batch, refine_loss_batch, valid_loss, similarity_matrix_batch = self.fusion(coords=up_coords, 
                                                                                                                        tsdf_target=tsdf_target, 
                                                                                                                        inputs=inputs, 
                                                                                                                        batch_id=batch, 
                                                                                                                        do_ins_fusion=True, 
                                                                                                                        ins_infos=ins_pred_batch_level,
                                                                                                                        loss_type=fusion_mode,  
                                                                                                                        )

            refined_masks.append(refined_mask)
            contrastive_losses.append(contrastive_loss_batch)
            refine_losses.append(refine_loss_batch)
            valid_losses.append(valid_loss)
            similarity_matrices.append(similarity_matrix_batch)  # [0, 1]

        if sum(valid_losses) > 0:  # valid loss
            loss_dict['contrastive'] = sum(contrastive_losses) / sum(valid_losses)
            loss_dict['refine'] = sum(refine_losses) / sum(valid_losses)
        else:
            loss_dict['contrastive'] = torch.zeros((1)).to(device)
            loss_dict['refine'] = torch.zeros((1)).to(device)

        "**********************   3D instance fusion   **********************"
        # fusion_masks = []
        # ins_masks = ins_pred_batch_level[batch]['img_level']
        # for k in range(len(ins_masks)):
        #     if len(ins_masks[k][1]) > 0:
        #         fusion_masks.append(ins_masks[k][1])
        # ins_masks = torch.cat(fusion_masks, dim=0)  # [n_SEG, nV]
        # ins_masks = ins_masks.sum(0) > 0

        # label2mesh(up_coords_batch[:, 1:], tsdf_target[batch_ind], tuple(self.cfg.N_VOX), ins_masks.float().unsqueeze(1), f'results/label_target_fusion.ply')

        return outputs, loss_dict, sum(valid_losses) > 0


    @staticmethod
    def compute_loss(tsdf, occ, tsdf_target, occ_target, loss_weight=(1, 1),
                     mask=None, pos_weight=1.0):
        '''

        :param tsdf: (Tensor), predicted tsdf, (N, 1)
        :param occ: (Tensor), predicted occupancy, (N, 1)
        :param tsdf_target: (Tensor),ground truth tsdf, (N, 1)
        :param occ_target: (Tensor), ground truth occupancy, (N, 1)
        :param loss_weight: (Tuple)
        :param mask: (Tensor), mask voxels which cannot be seen by all views
        :param pos_weight: (float)
        :return: loss: (Tensor)
        '''
        # compute occupancy/tsdf loss
        tsdf = tsdf.view(-1)
        occ = occ.view(-1)
        tsdf_target = tsdf_target.view(-1)
        occ_target = occ_target.view(-1)
        if mask is not None:
            mask = mask.view(-1)
            tsdf = tsdf[mask]
            occ = occ[mask]
            tsdf_target = tsdf_target[mask]
            occ_target = occ_target[mask]

        n_all = occ_target.shape[0]
        n_p = occ_target.sum()
        if n_p == 0:
            logger.warning('target: no valid voxel when computing loss')
            return torch.Tensor([0.0]).cuda()[0] * tsdf.sum()
        w_for_1 = (n_all - n_p).float() / n_p
        w_for_1 *= pos_weight

        # compute occ bce loss
        occ_loss = F.binary_cross_entropy_with_logits(occ, occ_target.float(), pos_weight=w_for_1)

        # compute tsdf l1 loss
        tsdf = apply_log_transform(tsdf[occ_target])
        tsdf_target = apply_log_transform(tsdf_target[occ_target])
        tsdf_loss = torch.mean(torch.abs(tsdf - tsdf_target))

        # compute final loss
        loss = loss_weight[0] * occ_loss + loss_weight[1] * tsdf_loss
        return loss


def tsdf2mesh(coords, tsdf, dim_list, save_path):
    tsdf = tsdf.view(-1).detach()
    tsdf_volume = sparse_to_dense_torch(coords.long(), tsdf, dim_list, 100, tsdf.device, final_tsdf=True)  
    tsdf_volume = tsdf_volume.cpu().numpy()

    verts, faces, norms, vals = measure.marching_cubes(tsdf_volume, level=0) 
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms)
    mesh.export(save_path)

def label2mesh(coords, tsdf, dim_list, label, save_path):
    tsdf = tsdf.view(-1).detach()
    tsdf_volume = sparse_to_dense_torch(coords.long(), tsdf, dim_list, 100, tsdf.device, final_tsdf=True)  
    tsdf_volume = tsdf_volume.cpu().numpy()

    label = label.view(-1).detach()
    label_volume = sparse_to_dense_torch(coords.long(), label, dim_list, 0, label.device, final_tsdf=False) 
    label_volume = label_volume.cpu().numpy()

    # Step 1: extract mesh
    verts, faces, norms, vals = measure.marching_cubes(tsdf_volume, level=0)  
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms)

    # Step 2: Map labels to mesh vertices
    # Use nearest neighbor interpolation to find the corresponding coordinates of each vertex in label_volume and get the label
    verts_scaled = (verts / np.array(tsdf_volume.shape)) * np.array(label_volume.shape)  
    verts_indices = np.round(verts_scaled).astype(int)  
    verts_indices = np.clip(verts_indices, 0, np.array(label_volume.shape) - 1) 
    vertex_labels = label_volume[verts_indices[:, 0], verts_indices[:, 1], verts_indices[:, 2]]

    # Step 3: Assign a color to each label
    unique_labels = np.unique(vertex_labels)
    colormap = cm.get_cmap('tab20', len(unique_labels))  
    label_to_color = {}
    for i, label in enumerate(unique_labels):
        if label == 0:
            label_to_color[label] = np.array([192, 192, 192], dtype=np.uint8)  
        else:
            label_to_color[label] = (np.array(colormap(i)[:3]) * 255).astype(np.uint8)

    vertex_colors = np.array([label_to_color[label] for label in vertex_labels])

    # Step 4
    mesh.visual.vertex_colors = vertex_colors
    mesh.export(save_path)

    print("mesh save to ", save_path)


def sparse_to_dense_torch(locs, values, dim, default_val, device, final_tsdf=False):

    if final_tsdf:  
        # default_val = 100
        # dense = torch.full([dim[0], dim[1], dim[2]], default_val, device=device).to(values.dtype)
        # if locs.shape[0] > 0:
        #     dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values
        #
        # mask = dense == default_val  
        #
        # dense, mask = dense.cpu().numpy(), mask.cpu().numpy()
        # distance, indices = ndimage.distance_transform_edt(mask, return_indices=True)  
        # nearest_values = dense[tuple(indices)]  
        # dense[mask] = np.where(nearest_values[mask] >= 0, 1, -1)  
        # dense = torch.from_numpy(dense).to(device)

        default_val = 1
        dense = torch.full([dim[0], dim[1], dim[2]], default_val, device=device).to(values.dtype)
        if locs.shape[0] > 0:
            dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values

    else:
        dense = torch.full([dim[0], dim[1], dim[2]], float(default_val), device=device).to(values.dtype)
        if locs.shape[0] > 0:
            dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values

    return dense

def save_mask(mask, save_path):
    mask_np = mask.detach().cpu().numpy() 
    
    mask_np = (mask_np * 255).astype(np.uint8)
    
    image = Image.fromarray(mask_np)
    image.save(save_path) 

def generate_mesh(coords, input, dim_list, save_path=None, mode='tsdf', export_mesh=False):
    if mode == 'tsdf':
        default_val = 1
        input = input.view(-1).detach()
        tsdf_volume = sparse_to_dense_torch(coords.long(), input, dim_list, default_val, input.device, final_tsdf=True)
        tsdf_volume = tsdf_volume.cpu().numpy()

        verts, faces, norms, vals = measure.marching_cubes(tsdf_volume, level=0)
        if export_mesh:
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms)
            mesh.export(save_path)

        return verts

    elif mode == 'panoptic':
        if isinstance(input, dict):
            target_label = input['labels'].int()
            target_instance = torch.arange(1, len(target_label) + 1).int().to(target_label.device)
            target_mask = input['masks'].int()
            first_true_indices = torch.argmax(target_mask, dim=0)
            # panoptic_mask = target_label[first_true_indices]
            panoptic_mask = target_instance[first_true_indices]
            panoptic_mask = panoptic_mask.detach().cpu().numpy()

        else:
            panoptic_mask = input.detach().cpu().numpy()

        # visualize_mesh(coords.to(torch.int32).detach().cpu().numpy(), panoptic_mask, type='instance')

        save_points_with_colors_to_ply(coords.to(torch.int32).detach().cpu().numpy(), panoptic_mask, save_path)  # point format
        # voxel_to_mesh_and_save(coords.to(torch.int32).detach().cpu().numpy(), panoptic_mask, save_path, colormap_name='viridis')  # mesh format

def save_points_with_colors_to_ply(points, labels, output_file):
    predefined_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
        (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
        (0, 128, 128), (128, 0, 128), (192, 192, 192), (128, 128, 128), (64, 64, 64),
        (255, 128, 0), (128, 255, 0), (0, 255, 128), (128, 0, 255), (255, 0, 128)
    ]
    num_colors = len(predefined_colors)

    label_to_color = {label: predefined_colors[i % num_colors] for i, label in enumerate(np.unique(labels))}

    colors = np.array([label_to_color[label] for label in labels])

    vertex_data = np.array(
        [(points[i, 0], points[i, 1], points[i, 2], colors[i, 0], colors[i, 1], colors[i, 2]) for i in range(len(points))],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    )

    vertex = PlyElement.describe(vertex_data, 'vertex')

    ply = PlyData([vertex])
    ply.write(output_file)


def voxel_to_mesh_and_save(points, labels, output_file='output_mesh.ply', colormap_name='viridis'):
    """
    Convert the voxel point set to a grid and save it as a PLY file.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    
    unique_labels = np.unique(labels)
    num_colors = len(unique_labels)
    colormap = cm.get_cmap(colormap_name, num_colors)
    label_to_color = {label: (np.array(colormap(i)[:3]) * 255).astype(np.uint8) for i, label in enumerate(unique_labels)}

    colors = np.array([label_to_color[label] for label in labels])
    poisson_mesh.vertex_colors = o3d.utility.Vector3dVector(colors / 255.0)  

    o3d.io.write_triangle_mesh(output_file, poisson_mesh)
