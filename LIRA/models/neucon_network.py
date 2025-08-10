import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsparse.tensor import PointTensor
from loguru import logger
import re
from copy import deepcopy
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
from scipy.ndimage import label, generate_binary_structure
import multiprocessing

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
from datasets.visualization import label2mesh
from tools.evaluation_3d import save_ply
from models.modules import keep_connected_region_3d

class NeuConNet(nn.Module):
    '''
    Coarse-to-fine network.
    '''

    def __init__(self, cfg):
        super(NeuConNet, self).__init__()
        self.cfg = cfg
        self.n_scales = len(cfg.THRESHOLDS) - 1

        channels = [96, 48, 24]  # (Features converted from fusion to prediction) -> The larger the resolution, the fewer feature dimensions should be

        self.selected_img_ids = cfg.SELECTED_IMG_IDS

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
        # for param in self.reason_seg_model.parameters():
        #     param.requires_grad = False

        # self.qa_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/grounding_scene_qa_infos_base/'
        # self.qa_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/grounding_scene_qa_infos_1/'
        
        # self.qa_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/grounding_scene_qa_infos_base_new/'
        # self.qa_path_all = ['/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/grounding_scene_qa_infos_base_new/', 
        #                     '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/grounding_scene_qa_infos_hard/']
        self.qa_path_all = ['/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/grounding_scene_qa_infos_base_new/']

        # self.qa_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/grounding_scene_qa_infos_hard/'
        self.mapping_save_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/grounding_scene_instance_infos_mapping/'


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

    def get_panoptic_targets(self, coords, inputs, scale, batch, qa_candidates):
        # valid_semantic_labels = [3, 4, 5, 6, 7, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36]
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
                if unique_id not in qa_candidates:
                    continue

                unique_indice = torch.where(instance_label == unique_id)[0]
                cur_label = torch.argmax(torch.bincount(semantic_label[unique_indice].to(torch.int)))
                # if cur_label not in valid_semantic_labels:
                #     continue

                labels.append(cur_label) # In case instance labels and category labels are not equal, select the most
                masks.append((instance_label == unique_id).unsqueeze(0))
                ins_ids_all.append(unique_id)

            if len(labels) != 0:
                panoptic_targets['labels'] = torch.tensor(labels).to(instance_target.device)  # (num_instance, )
                panoptic_targets['masks'] = torch.cat(masks, dim=0)  # (num_instance, num_voxel)
                panoptic_targets['ins_ids'] = torch.tensor(ins_ids_all).to(instance_target.device).to(labels[0].dtype)  # (num_instance, )

                del labels, masks

            return panoptic_targets

    def get_qa(self, inputs, batch):
        supplement_instruction = " Please give all objects that are helpful in inferring final targets, as well as the category and color of these objects."

        for self.qa_path in self.qa_path_all:
            qa_path = self.qa_path + str(inputs['scene'][batch]) + "_" + str(inputs['instruction_id'][batch].tolist()) + "_qa.pkl"
            if os.path.exists(qa_path):
                break

        with open(qa_path, 'rb') as file:
            scene_qa = pickle.load(file)
            qa = scene_qa[0]
            qa['instruction'] = qa['instruction'] + supplement_instruction + ' Please output segmentation masks.'

        with open(self.mapping_save_path + str(inputs['scene'][batch]) + '.pkl', 'rb') as file:
            ins_mapping_dict = pickle.load(file)

            for i in range(len(qa['instance ids'])):
                qa['instance ids'][i] = ins_mapping_dict[str(qa['instance ids'][i])]

            for j in range(len(qa['candidate'])):
                qa['candidate'][j] = ins_mapping_dict[str(qa['candidate'][j])]   

        qa_instruction = qa['instruction']
        qa_candidates = torch.tensor(qa['candidate'])

        return qa_instruction, qa_candidates

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

    # Run inference and set timeout
    # def run_inference_with_timeout(self, batch_indexs, prompt_all, rs_imgs_np_all, timeout=60):
    #     # Create a process to run inference
    #     def target(batch_indexs, prompt_all, rs_imgs_np_all):
    #         return self.reason_seg_model.inference_batch_parallel(batch_indexs, prompt_all, rs_imgs_np_all)

    #     process = multiprocessing.Process(target=target, args=(batch_indexs, prompt_all, rs_imgs_np_all))
    #     process.start()
    #     process.join(timeout)  

    #     if process.is_alive():
    #         print(f"Reason_seg exceeded the time limit ({timeout} seconds), skipping it.")
    #         process.terminate()  # Terminate the process after timeout
    #         return None, None, None, None  

    #     return target(batch_indexs, prompt_all, rs_imgs_np_all)

    def forward(self, inputs, outputs):
        # t_infer_start = time.time()

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
        valid_losses = [] 

        up_coords_all = []
        tsdf_target_all = []
        qa_instruction_all = []

        rs_imgs_np_all = []
        batch_indexs = []
        prompt_all = []

        for batch in range(bs):
            "***********************************************   geometry reconstruction   ************************************************"
            # ----   qa   ---- 
            qa_instruction, qa_candidates = self.get_qa(inputs, batch)
            qa_candidates = qa_candidates.to(device)
            # print("qa_instruction: ", qa_instruction)
            # print("qa_candidates: ", qa_candidates)

            up_coords, tsdf_target, occ_target = self.fusion(inputs=inputs, batch_id=batch)  # ScanNet v2 The camera field of view (FOV) of the dataset is horizontal field of view (70 degrees), vertical field of view (55 degrees)
            up_coords_all.append(up_coords)
            tsdf_target_all.append(tsdf_target) 
            qa_instruction_all.append(qa_instruction)
            # visualize panoptic target
            # for batch in range(bs):
            #     batch_ind = torch.nonzero(up_coords[:, 0] == batch).squeeze(1)
            #     generate_mesh(up_coords[batch_ind][:, 1:], panoptic_targets[batch], volume_shape, 'results/panoptic_target.ply', mode='panoptic')

            # ----   generate panoptic target   ---- 

            """
            Each batch has a list, and within each list:
            'labels': Instance category labels (including 40 categories of semantic labels)
            'masks': shape:[num_semantic_labels, num_voxels]
            'ins_ids': instance ids
            """
            panoptic_targets.append(self.get_panoptic_targets(up_coords, inputs, 0, batch, qa_candidates))

            ins_pred_batch_level.append({})

            '**************************   If there is no GT instance in the current FBV, skip fusion and refinement directly   **************************'
            # if len(panoptic_targets[batch]) == 0:
            #     valid_losses.append(False)
            #     continue
            # isin_gt = torch.isin(panoptic_targets[batch]['ins_ids'], qa_candidates)
            # if not isin_gt.any():
            #     valid_losses.append(False)
            #     continue

            "***********************************************   reason_seg start   ************************************************"
            imgs = inputs['imgs'][batch]  # [N_views, 3, H, W] = [9, 3, 480, 640]

            reason_seg_imgs = imgs.detach().cpu() 
            reason_seg_imgs = reason_seg_imgs / 255  # 0 ~ 1

            # # vis as image
            # to_pil = transforms.ToPILImage()
            # for vis_i in range(reason_seg_imgs.size(0)):
            #     vis_img = reason_seg_imgs[vis_i]
            #     vis_img = to_pil(vis_img) 
            #     vis_img.save(f"image_{vis_i}.png")  

            for rs_i in range(reason_seg_imgs.size(0)):
                rs_imgs_np = reason_seg_imgs[rs_i].permute(1, 2, 0).numpy()  
                rs_imgs_np = (rs_imgs_np * 255).astype(np.uint8) 
                ori_height, ori_width, _ = rs_imgs_np.shape  
                resized_height, resized_width = int(ori_height / 4), int(ori_width / 4)  # features: list9: list3: 1/4-[1, 24, 120, 160], 1/8-[1, 40, 60, 80], 1/16-[1, 80, 30, 40]
                rs_imgs_np = cv2.resize(rs_imgs_np, (1024, 1024))
                rs_imgs_np_all.append(rs_imgs_np)

            batch_indexs.append(batch)
            prompt_all.append(qa_instruction)
        
        if len(batch_indexs) > 0:
            "reason seg Multi-batch parallel inference [b*9, 3, 480, 640]"
            with torch.no_grad():
                # masks: [N_mask, 1024, 1024], rs_text_feats: [N_mask, 4096], rs_img_feats: [N_mask, 256, 1024, 1024]
                # t_reason_seg_start = time.time()
                response_all, masks_all, rs_text_feats_all, rs_img_feats_all = self.reason_seg_model.inference_batch_parallel(batch_indexs, prompt_all, rs_imgs_np_all)  # Reason_seg performs parallel reasoning on N images
                # response_all, masks_all, rs_text_feats_all, rs_img_feats_all = self.run_inference_with_timeout(batch_indexs, prompt_all, rs_imgs_np_all, timeout=60)  # Timeout processing(s)
                # print('Reason Seg inference time: {:.2f} s'.format(time.time() - t_reason_seg_start), '*' * 100)

            if response_all is not None:
                rs_img_feats_all = rs_img_feats_all.to(torch.float32)  # float32 contains all the value ranges of bf16
                rs_img_feats_all = F.interpolate(rs_img_feats_all, size=(resized_height, resized_width), mode='bilinear', align_corners=False)

                batch_index_count = -1
                for batch in range(bs):
                    if not batch in batch_indexs:
                        continue
                    
                    batch_index_count += 1
                    up_coords_batch = up_coords_all[batch]

                    " Collect each image "
                    ins_pred_img_level = []
                    occ_feats_volume_batch = []  # Project the features of the 9 images into the voxel space and take the average
                    for rs_i in range(len(self.selected_img_ids)):  # reason_seg_imgs: [9, 3, 480, 640]
                        batch_rs_i = batch_index_count * len(self.selected_img_ids) + rs_i

                        masks = masks_all[batch_rs_i].float().unsqueeze(1)
                        masks = F.interpolate(masks, size=(resized_height, resized_width), mode='nearest')  # Optional mode: 'nearest' corresponds to nearest neighbor interpolation
                        masks = masks.squeeze(1).bool() 
                        # masks = keep_largest_connected_region_batch(masks)  # Eliminate discrete small areas (Connected Component Analysis)
                        # save_mask(masks[0], "mask_bw.png")  

                        rs_img_feats = rs_img_feats_all[batch_rs_i].unsqueeze(0)

                        # back-project to voxel
                        ins_text = [match.strip() for match in re.findall(r'([\w\s]+)\s\[SEG\]', response_all[batch_rs_i])]
                        ins_volume = []  # The instance category corresponding to each voxel [num_ins, num_voxels]

                        depth = inputs['depth'][batch][self.selected_img_ids[rs_i]].unsqueeze(0).unsqueeze(0)
                        depth = F.interpolate(depth, size=(resized_height, resized_width), mode='nearest')
                        depth = depth.squeeze(0).squeeze(0)
                        if (depth > 0).sum() < 50:
                            continue
                        
                        for m in range(len(masks)):
                            ins_volume.append(rs_ins_back_project(up_coords_batch[:, 1:], inputs['vol_origin_partial'][batch], self.cfg.VOXEL_SIZE,  
                                                                masks[m].unsqueeze(0), KRcam[:, batch], self.selected_img_ids[rs_i], depth)) 
                        if len(ins_text) != len(ins_volume):  # If the decoded text does not match the number of masks, skip
                            continue
                        occ_feats_volume_batch.append(rs_ins_back_project(up_coords_batch[:, 1:], inputs['vol_origin_partial'][batch], self.cfg.VOXEL_SIZE, 
                                                                        rs_img_feats.squeeze(0), KRcam[:, batch], self.selected_img_ids[rs_i], depth))  

                        # **********************************************   vis   **********************************************
                        # # save_depth(inputs['depth'][batch][self.selected_img_ids[rs_i]].cpu().numpy(), f'depth_{batch_rs_i}.png')  # save depth as heatmap

                        # if len(ins_volume) == 0:
                        #     mask_labels = torch.zeros_like(up_coords_batch[:, 1]).to(torch.int64)
                        # else:
                        #     mask_labels = transform_mask(ins_volume)
                        # # Take the largest connected area
                        # mask_labels = keep_connected_region_3d(up_coords_batch[:, 1:], mask_labels, n=2)  
                        
                        # label2mesh(up_coords_batch[:, 1:], tsdf_target_all[batch], tuple(self.cfg.N_VOX), mask_labels.float().unsqueeze(1), f'results/label_target_{batch_rs_i}.ply')
                        # # save_ply(up_coords_batch[:, 1:].cpu().numpy(), mask_labels.cpu().numpy(), f'results/label_target_{batch_rs_i}.ply')

                        """
                        For each image, assign each instance to an id:
                        list 0: (num_voxels, 3), coodinates
                        list 1: (num_instance, num_voxels), The mask corresponding to each ins, bool type
                        list 2: 
                            list 0: dict: {'id': 1, 'category': 'black chair', 'sem_id': sem class}
                            ...
                        list 3: (num_instance, text_feat_dim), Text feats corresponding to each instance
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
                            if ins_volume[m].sum() >= 200:  # The 2D mask mapping is considered valid only if it has more than 1000 voxels (5%)! (120*160=19200)
                                ins_preds[1].append(ins_volume[m])
                                ins_preds[2].append({'id': ins_id_count, 'category': ins_text[m]})
                                ins_preds[3].append(rs_text_feats_all[batch_rs_i][m].to(torch.float32))
                                ins_preds[4].append(rs_i)
                                ins_id_count += 1

                        if len(ins_preds[1]) > 0:
                            ins_preds[0] = up_coords_batch[:, 1:]
                            ins_preds[1] = torch.stack(ins_preds[1])
                            ins_preds[3] = torch.stack(ins_preds[3])
                            ins_preds[4] = torch.tensor(ins_preds[4]).to(ins_preds[3].device)

                        ins_pred_img_level.append(ins_preds)

                    # Fuse the features of 9 images into occ_feats_volume
                    occ_feats_volume_batch = torch.stack(occ_feats_volume_batch)  # [9, N, 256]
                    occ_mask_batch = torch.all(occ_feats_volume_batch == 0, dim=2)  # [9, N]
                    occ_mask_batch = ~occ_mask_batch
                    occ_mask_batch = occ_mask_batch.sum(dim=0)  # [N,] The number of image features that each voxel can see
                    occ_mask_batch[occ_mask_batch == 0] = 1  # Prevent division by zero

                    occ_feats_volume_b = occ_feats_volume_batch.sum(dim=0)  # [N, 256]
                    occ_feats_volume_b = occ_feats_volume_b / occ_mask_batch.unsqueeze(1) 

                    ins_pred_batch_level[batch]['img_level'] = ins_pred_img_level
                    ins_pred_batch_level[batch]['occ_level'] = occ_feats_volume_b

                    del occ_mask_batch, occ_feats_volume_b

                    # update targets
                    occ_mask_batch = torch.all(occ_feats_volume_batch == 0, dim=2)
                    occ_mask_batch = ~occ_mask_batch
                    contrastive_iou_mask = []

                    for k in range(len(occ_mask_batch)):
                        contrastive_iou_mask.append(occ_mask_batch[k].unsqueeze(0).repeat(len(ins_pred_img_level[k][2]), 1))
                    contrastive_iou_mask = torch.cat(contrastive_iou_mask, dim=0)  # [n_SEG, nV]

                    panoptic_targets[batch]['contrastive_iou_mask'] = contrastive_iou_mask
                    ins_pred_batch_level[batch]['gt'] = panoptic_targets[batch]

                    # 3D instance fusion
                    contrastive_loss_batch, valid_loss = self.fusion(coords_target=up_coords_batch, 
                                                                    tsdf_target=tsdf_target_all[batch], 
                                                                    inputs=inputs, 
                                                                    batch_id=batch, 
                                                                    do_ins_fusion=True, 
                                                                    ins_infos=ins_pred_batch_level, 
                                                                    qa_instruction=qa_instruction_all[batch], 
                                                                    )
                
                    contrastive_losses.append(contrastive_loss_batch)
                    valid_losses.append(valid_loss)

        if sum(valid_losses) > 0:  # has valid loss
            loss_dict['contrastive'] = sum(contrastive_losses) / sum(valid_losses)
        else:
            fake_sample = torch.ones((1, 128), device=device)
            loss_dict['contrastive'] = self.fusion.confidence_transform(fake_sample) * 0

        # t_infer_end = time.time()
        # print("infer time: {:.2f}".format(t_infer_end-t_infer_start), '*' * 100)

        "vis global fused map"
        # vis_coords = self.fusion.target_tsdf_volume[2].C - self.fusion.target_tsdf_volume[2].C.min(dim=0)[0]
        # vis_scope = vis_coords.max(dim=0)[0].cpu().tolist() 
        # vis_scope = [vis_x + 10 for vis_x in vis_scope]
        # label2mesh(vis_coords, 
        #            self.fusion.target_tsdf_volume[2].F.squeeze(1), 
        #            tuple(vis_scope), 
        #            self.fusion.global_instance.unsqueeze(1), 
        #            f'results/global_fused_mesh.ply')

        torch.cuda.empty_cache()
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

def keep_largest_connected_region_batch(mask: torch.Tensor) -> torch.Tensor:
    """
    The largest connected region in each mask is retained and other small regions are removed.
    """
    
    N, H, W = mask.shape  

    max_area_mask = torch.zeros_like(mask)

    # Process each mask one by one
    for i in range(N):
        mask_np = mask[i].cpu().numpy()  
        
        structure = generate_binary_structure(2, 2)  # 8-Connectivity

        labeled_mask, num_labels = label(mask_np, structure)

        region_sizes = np.bincount(labeled_mask.ravel())

        max_label = region_sizes[1:].argmax() + 1  

        max_area_mask_np = (labeled_mask == max_label)

        max_area_mask[i] = torch.from_numpy(max_area_mask_np).to(mask.dtype).to(mask.device)

    return max_area_mask

def tsdf2mesh(coords, tsdf, dim_list, save_path):
    tsdf = tsdf.view(-1).detach()
    tsdf_volume = sparse_to_dense_torch(coords.long(), tsdf, dim_list, 100, tsdf.device, final_tsdf=True)  # The final tsdf needs to be filled with -1 and 1
    tsdf_volume = tsdf_volume.cpu().numpy()

    verts, faces, norms, vals = measure.marching_cubes(tsdf_volume, level=0)  
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms)
    mesh.export(save_path)

def sparse_to_dense_torch(locs, values, dim, default_val, device, final_tsdf=False):

    if final_tsdf:  # The final tsdf needs to be filled with -1 and 1
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
    """
    Saves a set of spatial 3D points and their corresponding semantic labels as a PLY file, assigning each point a distinct color.
    If the number of labels exceeds 20, the colors are cycled from the beginning.

    参数：
    points (numpy.ndarray): 
    labels (numpy.ndarray): semantic labels
    output_file (str): 

    返回：
    None
    """
    predefined_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
        (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
        (0, 128, 128), (128, 0, 128), (192, 192, 192), (128, 128, 128), (64, 64, 64),
        (255, 128, 0), (128, 255, 0), (0, 255, 128), (128, 0, 255), (255, 0, 128)
    ]
    num_colors = len(predefined_colors)

    label_to_color = {label: predefined_colors[i % num_colors] for i, label in enumerate(np.unique(labels))}\
    
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

    params：
    points (numpy.ndarray): (N, 3)
    labels (numpy.ndarray): semantic labels
    output_file (str): 
    colormap_name (str): 

    return：
    None
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

def save_depth(depth: np.ndarray, output_path: str):
    """
    Normalize the depth map to 0-1 and save it as a heatmap.

    参数:
        depth (np.ndarray): depth map
        output_path (str): 
    """
    if depth.size == 0:
        raise ValueError("input depth map is none")

    depth_min = np.min(depth)
    depth_max = np.max(depth)

    if depth_max > depth_min:  
        depth_normalized = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_normalized = np.zeros_like(depth) 

    depth_colormap = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)

    cv2.imwrite(output_path, depth_colormap)


def transform_mask(instance_infos):

    l = len(instance_infos)
    if l == 0:
        return None
    
    else:
        label = torch.zeros_like(instance_infos[0]).to(torch.int64)
        for i in range(len(instance_infos)):
            label[instance_infos[i]] = i + 1

        return label
