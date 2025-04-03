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

        channels = [96, 48, 24]  # (融合后转换到预测之前的特征) -> 分辨率越大，特征维度应该越少

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

    # 运行推理并设置超时
    # def run_inference_with_timeout(self, batch_indexs, prompt_all, rs_imgs_np_all, timeout=60):
    #     # 创建一个进程来运行推理
    #     def target(batch_indexs, prompt_all, rs_imgs_np_all):
    #         return self.reason_seg_model.inference_batch_parallel(batch_indexs, prompt_all, rs_imgs_np_all)

    #     process = multiprocessing.Process(target=target, args=(batch_indexs, prompt_all, rs_imgs_np_all))
    #     process.start()
    #     process.join(timeout)  # 等待进程最多 `timeout` 秒

    #     if process.is_alive():
    #         print(f"Reason_seg exceeded the time limit ({timeout} seconds), skipping it.")
    #         process.terminate()  # 超时后终止进程
    #         return None, None, None, None  # 返回 None 表示超时

    #     # 如果进程正常完成，返回推理结果
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

            up_coords, tsdf_target, occ_target = self.fusion(inputs=inputs, batch_id=batch)  # ScanNet v2 数据集的相机视场(FOV)为 水平视场(70度)、垂直视场(55度)
            up_coords_all.append(up_coords)
            tsdf_target_all.append(tsdf_target) 
            qa_instruction_all.append(qa_instruction)
            # 可视化panoptic target
            # for batch in range(bs):
            #     batch_ind = torch.nonzero(up_coords[:, 0] == batch).squeeze(1)
            #     generate_mesh(up_coords[batch_ind][:, 1:], panoptic_targets[batch], volume_shape, 'results/panoptic_target.ply', mode='panoptic')

            # ----   generate panoptic target   ---- 

            """
            每个batch一个列表, 每个列表内:
            'labels': 实例的类别标签(包含40类语义标签)
            'masks': shape:[num_semantic_labels, num_voxels]
            'ins_ids': 实例的id
            """
            panoptic_targets.append(self.get_panoptic_targets(up_coords, inputs, 0, batch, qa_candidates))

            ins_pred_batch_level.append({})

            '**************************   如果当前FBV内部没有GT实例, 直接跳过fusion和refinement   **************************'
            # if len(panoptic_targets[batch]) == 0:
            #     valid_losses.append(False)
            #     continue
            # isin_gt = torch.isin(panoptic_targets[batch]['ins_ids'], qa_candidates)
            # if not isin_gt.any():
            #     valid_losses.append(False)
            #     continue

            "***********************************************   reason_seg start   ************************************************"
            imgs = inputs['imgs'][batch]  # [N_views, 3, H, W] = [9, 3, 480, 640]

            # 可视化成图片
            reason_seg_imgs = imgs.detach().cpu() 
            reason_seg_imgs = reason_seg_imgs / 255  # 0 ~ 1

            # # 可视化成图片
            # to_pil = transforms.ToPILImage()
            # for vis_i in range(reason_seg_imgs.size(0)):
            #     vis_img = reason_seg_imgs[vis_i]
            #     vis_img = to_pil(vis_img)  # 转换为 PIL 图片
            #     vis_img.save(f"image_{vis_i}.png")  

            for rs_i in range(reason_seg_imgs.size(0)):
                rs_imgs_np = reason_seg_imgs[rs_i].permute(1, 2, 0).numpy()  # 转换为 NumPy 格式 (H, W, C) 并从 [0, 1] 归一化到 [0, 255]
                rs_imgs_np = (rs_imgs_np * 255).astype(np.uint8)  # 转为 uint8 格式
                ori_height, ori_width, _ = rs_imgs_np.shape  # 高、宽、通道数  (480, 640)
                resized_height, resized_width = int(ori_height / 4), int(ori_width / 4)  # 最大特征图为/4;  features: list9: list3: 1/4-[1, 24, 120, 160], 1/8-[1, 40, 60, 80], 1/16-[1, 80, 30, 40]
                rs_imgs_np = cv2.resize(rs_imgs_np, (1024, 1024))
                rs_imgs_np_all.append(rs_imgs_np)

            batch_indexs.append(batch)
            prompt_all.append(qa_instruction)
        
        if len(batch_indexs) > 0:
            "reason seg 多batch并行推理 [b*9, 3, 480, 640]"
            with torch.no_grad():
                # masks: [N_mask, 1024, 1024], rs_text_feats: [N_mask, 4096], rs_img_feats: [N_mask, 256, 1024, 1024]
                # t_reason_seg_start = time.time()
                response_all, masks_all, rs_text_feats_all, rs_img_feats_all = self.reason_seg_model.inference_batch_parallel(batch_indexs, prompt_all, rs_imgs_np_all)  # reason_seg对N张图片并行推理
                # response_all, masks_all, rs_text_feats_all, rs_img_feats_all = self.run_inference_with_timeout(batch_indexs, prompt_all, rs_imgs_np_all, timeout=60)  # 超时处理(s)
                # print('Reason Seg inference time: {:.2f} s'.format(time.time() - t_reason_seg_start), '*' * 100)

            if response_all is not None:
                rs_img_feats_all = rs_img_feats_all.to(torch.float32)  # float32包含bf16所有的数值范围
                rs_img_feats_all = F.interpolate(rs_img_feats_all, size=(resized_height, resized_width), mode='bilinear', align_corners=False)

                batch_index_count = -1
                for batch in range(bs):
                    if not batch in batch_indexs:
                        continue
                    
                    batch_index_count += 1
                    up_coords_batch = up_coords_all[batch]

                    " 对每张图片进行收集 "
                    ins_pred_img_level = []
                    occ_feats_volume_batch = []  # 将9张图像的特征投影到voxel空间，并取平均
                    for rs_i in range(len(self.selected_img_ids)):  # reason_seg_imgs: [9, 3, 480, 640]
                        batch_rs_i = batch_index_count * len(self.selected_img_ids) + rs_i

                        # resize回原始尺寸
                        masks = masks_all[batch_rs_i].float().unsqueeze(1)
                        masks = F.interpolate(masks, size=(resized_height, resized_width), mode='nearest')  # 可选 mode: 'nearest' 对应最近邻插值
                        masks = masks.squeeze(1).bool() 
                        # masks = keep_largest_connected_region_batch(masks)  # 剔除离散小区域 (连通区域分析 Connected Component Analysis)
                        # save_mask(masks[0], "mask_bw.png")  # 保存mask

                        rs_img_feats = rs_img_feats_all[batch_rs_i].unsqueeze(0)

                        # back-project to voxel
                        ins_text = [match.strip() for match in re.findall(r'([\w\s]+)\s\[SEG\]', response_all[batch_rs_i])]
                        ins_volume = []  # 每个voxel对应实例类别 [num_ins, num_voxels]

                        depth = inputs['depth'][batch][self.selected_img_ids[rs_i]].unsqueeze(0).unsqueeze(0)
                        depth = F.interpolate(depth, size=(resized_height, resized_width), mode='nearest')
                        depth = depth.squeeze(0).squeeze(0)
                        if (depth > 0).sum() < 50:
                            continue
                        
                        for m in range(len(masks)):
                            ins_volume.append(rs_ins_back_project(up_coords_batch[:, 1:], inputs['vol_origin_partial'][batch], self.cfg.VOXEL_SIZE,  
                                                                masks[m].unsqueeze(0), KRcam[:, batch], self.selected_img_ids[rs_i], depth)) 
                        if len(ins_text) != len(ins_volume):  # 如果解码出的文本与mask数量不一致，跳过
                            continue
                        occ_feats_volume_batch.append(rs_ins_back_project(up_coords_batch[:, 1:], inputs['vol_origin_partial'][batch], self.cfg.VOXEL_SIZE, 
                                                                        rs_img_feats.squeeze(0), KRcam[:, batch], self.selected_img_ids[rs_i], depth))  

                        # **********************************************   vis   **********************************************
                        # # save_depth(inputs['depth'][batch][self.selected_img_ids[rs_i]].cpu().numpy(), f'depth_{batch_rs_i}.png')  # save depth as heatmap

                        # if len(ins_volume) == 0:
                        #     mask_labels = torch.zeros_like(up_coords_batch[:, 1]).to(torch.int64)
                        # else:
                        #     mask_labels = transform_mask(ins_volume)
                        # # 取最大连通区域
                        # mask_labels = keep_connected_region_3d(up_coords_batch[:, 1:], mask_labels, n=2)  
                        
                        # label2mesh(up_coords_batch[:, 1:], tsdf_target_all[batch], tuple(self.cfg.N_VOX), mask_labels.float().unsqueeze(1), f'results/label_target_{batch_rs_i}.ply')
                        # # save_ply(up_coords_batch[:, 1:].cpu().numpy(), mask_labels.cpu().numpy(), f'results/label_target_{batch_rs_i}.ply')

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
                            if ins_volume[m].sum() >= 200:  # 2D的mask映射大于1000个voxel(5%), 才认为是有效的！ (120*160=19200)
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

                    # 将9张图片的特征融合到occ_feats_volume
                    occ_feats_volume_batch = torch.stack(occ_feats_volume_batch)  # [9, N, 256]
                    occ_mask_batch = torch.all(occ_feats_volume_batch == 0, dim=2)  # [9, N]
                    occ_mask_batch = ~occ_mask_batch
                    occ_mask_batch = occ_mask_batch.sum(dim=0)  # [N,] 每个voxel能看到的图像特征数量
                    occ_mask_batch[occ_mask_batch == 0] = 1  # 防止除0

                    occ_feats_volume_b = occ_feats_volume_batch.sum(dim=0)  # [N, 256]
                    occ_feats_volume_b = occ_feats_volume_b / occ_mask_batch.unsqueeze(1)  # 特征均值

                    # 收集特征
                    ins_pred_batch_level[batch]['img_level'] = ins_pred_img_level
                    ins_pred_batch_level[batch]['occ_level'] = occ_feats_volume_b

                    del occ_mask_batch, occ_feats_volume_b

                    # 更新targets
                    occ_mask_batch = torch.all(occ_feats_volume_batch == 0, dim=2)
                    occ_mask_batch = ~occ_mask_batch
                    contrastive_iou_mask = []

                    for k in range(len(occ_mask_batch)):
                        contrastive_iou_mask.append(occ_mask_batch[k].unsqueeze(0).repeat(len(ins_pred_img_level[k][2]), 1))
                    contrastive_iou_mask = torch.cat(contrastive_iou_mask, dim=0)  # [n_SEG, nV]

                    panoptic_targets[batch]['contrastive_iou_mask'] = contrastive_iou_mask
                    ins_pred_batch_level[batch]['gt'] = panoptic_targets[batch]

                    # 3D实例融合
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

        if sum(valid_losses) > 0:  # 有合法loss
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
    保留每个掩码中的最大连通区域，去除其他小区域。
    
    参数:
        mask (torch.Tensor): 输入的二值掩码，形状为 (N, H, W)，类型为 torch.bool。
        
    返回:
        torch.Tensor: 只保留每个掩码中最大连通区域的掩码，形状为 (N, H, W)，类型为 torch.bool。
    """
    
    N, H, W = mask.shape  # 获取批次大小 N 和图像的高宽 H 和 W
    
    # 创建一个空的 tensor 来存储每个掩码的最大区域
    max_area_mask = torch.zeros_like(mask)

    # 逐个处理每个掩码
    for i in range(N):
        mask_np = mask[i].cpu().numpy()  # 获取第 i 个掩码
        
        # 生成连接结构元素（2D 图像，4连通或8连通）
        structure = generate_binary_structure(2, 2)  # 8-连通

        # 使用 scipy.ndimage.label 对掩码进行连通区域标记
        labeled_mask, num_labels = label(mask_np, structure)

        # 计算每个连通区域的面积（像素数）
        region_sizes = np.bincount(labeled_mask.ravel())

        # 找到最大面积的区域（排除背景区域，背景的标签为0）
        max_label = region_sizes[1:].argmax() + 1  # `argmax` 返回最大区域的索引，+1 是因为背景标签为0

        # 创建只保留最大区域的掩码
        max_area_mask_np = (labeled_mask == max_label)

        # 将结果保存回 max_area_mask
        max_area_mask[i] = torch.from_numpy(max_area_mask_np).to(mask.dtype).to(mask.device)

    return max_area_mask

def tsdf2mesh(coords, tsdf, dim_list, save_path):
    tsdf = tsdf.view(-1).detach()
    tsdf_volume = sparse_to_dense_torch(coords.long(), tsdf, dim_list, 100, tsdf.device, final_tsdf=True)  # 最后的tsdf需要用-1和1填充
    tsdf_volume = tsdf_volume.cpu().numpy()

    verts, faces, norms, vals = measure.marching_cubes(tsdf_volume, level=0)  # level: 等值面的标量值，函数将提取此标量值对应的表面
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms)
    mesh.export(save_path)

def sparse_to_dense_torch(locs, values, dim, default_val, device, final_tsdf=False):

    if final_tsdf:  # 最后的tsdf需要用最近邻-1和1填充
        # default_val = 100
        # dense = torch.full([dim[0], dim[1], dim[2]], default_val, device=device).to(values.dtype)
        # if locs.shape[0] > 0:
        #     dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values
        #
        # mask = dense == default_val  # 需要被换掉的值
        #
        # dense, mask = dense.cpu().numpy(), mask.cpu().numpy()
        # distance, indices = ndimage.distance_transform_edt(mask, return_indices=True)  # 计算距离变换，得到每个点到最近的不需要被替换点的距离
        # nearest_values = dense[tuple(indices)]  # 使用indices映射，找到每个点最近的不需要被替换的点的值
        # dense[mask] = np.where(nearest_values[mask] >= 0, 1, -1)  # 根据最近点的值来决定替换值，正数替换为1，负数替换为-1
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
    # 1. 将布尔张量转换为 NumPy 数组
    mask_np = mask.detach().cpu().numpy()  # 确保张量在 CPU 上
    
    # 2. 将布尔值映射到黑白值（True -> 255, False -> 0）
    mask_np = (mask_np * 255).astype(np.uint8)
    
    # 3. 保存为黑白图片
    image = Image.fromarray(mask_np)
    image.save(save_path)  # 保存为 PNG 文件

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
    将空间三维点集和对应的语义标签保存为 PLY 文件，并为每个点分配明显不同的颜色。
    如果标签数量超过 20, 则颜色从头循环。

    参数：
    points (numpy.ndarray): 形状为 (N, 3) 的三维点集
    labels (numpy.ndarray): 形状为 (N,) 的语义标签
    output_file (str): 输出的 PLY 文件路径

    返回：
    None
    """
    # 定义 20 种明显不同的颜色 (R, G, B)
    predefined_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
        (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
        (0, 128, 128), (128, 0, 128), (192, 192, 192), (128, 128, 128), (64, 64, 64),
        (255, 128, 0), (128, 255, 0), (0, 255, 128), (128, 0, 255), (255, 0, 128)
    ]
    num_colors = len(predefined_colors)

    # 为每个标签分配颜色，超过 20 的标签从头循环
    label_to_color = {label: predefined_colors[i % num_colors] for i, label in enumerate(np.unique(labels))}

    # 为每个点分配颜色
    colors = np.array([label_to_color[label] for label in labels])

    # 创建 PLY 文件数据结构
    vertex_data = np.array(
        [(points[i, 0], points[i, 1], points[i, 2], colors[i, 0], colors[i, 1], colors[i, 2]) for i in range(len(points))],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    )

    # 创建 PlyElement 对象
    vertex = PlyElement.describe(vertex_data, 'vertex')

    # 保存为 PLY 文件
    ply = PlyData([vertex])
    ply.write(output_file)

    # print(f"PLY 文件已保存至 {output_file}!")


def voxel_to_mesh_and_save(points, labels, output_file='output_mesh.ply', colormap_name='viridis'):
    """
    将体素点集转换为网格, 并保存为PLY文件。

    参数：
    points (numpy.ndarray): 形状为 (N, 3) 的三维体素点集
    labels (numpy.ndarray): 形状为 (N,) 的语义标签
    output_file (str): 输出的PLY文件路径
    colormap_name (str): 用于生成颜色的matplotlib colormap名称

    返回：
    None
    """
    # 将 NumPy 数组转为 open3d 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 计算法线
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # 使用 Poisson Surface Reconstruction 从点云生成网格
    poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    
    # 将网格的顶点颜色设置为标签的颜色（类似于之前的颜色映射）
    unique_labels = np.unique(labels)
    num_colors = len(unique_labels)
    colormap = cm.get_cmap(colormap_name, num_colors)
    label_to_color = {label: (np.array(colormap(i)[:3]) * 255).astype(np.uint8) for i, label in enumerate(unique_labels)}
    
    # 为每个顶点分配颜色
    colors = np.array([label_to_color[label] for label in labels])
    poisson_mesh.vertex_colors = o3d.utility.Vector3dVector(colors / 255.0)  # 将颜色归一化到 [0, 1] 之间

    # 保存为 PLY 文件
    o3d.io.write_triangle_mesh(output_file, poisson_mesh)
    # print(f"PLY 文件已保存至 {output_file}!")

def save_depth(depth: np.ndarray, output_path: str):
    """
    将深度图归一化到 0-1 并保存为热力图。

    参数:
        depth (np.ndarray): 深度图，形状为 (480, 640)。
        output_path (str): 保存热力图的文件路径，例如 "xxx.png"。
    """
    # 检查输入深度图是否为空
    if depth.size == 0:
        raise ValueError("输入深度图为空！")

    # 将深度图归一化到 0-1
    depth_min = np.min(depth)
    depth_max = np.max(depth)

    if depth_max > depth_min:  # 防止除以零
        depth_normalized = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_normalized = np.zeros_like(depth)  # 如果所有值相等，则返回全零图像

    # 将归一化的深度图转换为热力图（使用 OpenCV 的 COLORMAP_JET）
    depth_colormap = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # 保存热力图
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