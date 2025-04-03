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

from models.modules import SPVCNN
from neu_utils import apply_log_transform
from .gru_fusion import GRUFusion
from ops.back_project import back_project, ins_back_project, mask_reshape
from ops.generate_grids import generate_grid
from skimage import measure
import trimesh
from datasets.visualization import visualize_mesh
# from models.grounding_2d_llava_grounding import Grounding_2D


class NeuConNet(nn.Module):
    '''
    Coarse-to-fine network.
    '''

    def __init__(self, cfg):
        super(NeuConNet, self).__init__()
        self.cfg = cfg
        self.n_scales = len(cfg.THRESHOLDS) - 1

        alpha = int(self.cfg.BACKBONE2D.ARC.split('-')[-1])
        ch_in = [80 * alpha + 1, 96 + 40 * alpha + 2 + 1, 48 + 24 * alpha + 2 + 1, 24 + 24 + 2 + 1]  # [81, 139, 75, 51] (融合后的特征)
        channels = [96, 48, 24]  # (融合后转换到预测之前的特征) -> 分辨率越大，特征维度应该越少

        if self.cfg.FUSION.FUSION_ON:
            # GRU Fusion
            self.gru_fusion = GRUFusion(cfg, channels)
        # sparse conv
        self.sp_convs = nn.ModuleList()
        # MLPs that predict tsdf and occupancy.
        self.tsdf_preds = nn.ModuleList()
        self.occ_preds = nn.ModuleList()
        # for i in range(len(cfg.THRESHOLDS)):
        #     self.sp_convs.append(
        #         SPVCNN(num_classes=1, in_channels=ch_in[i],
        #                pres=1,
        #                cr=1 / 2 ** i,
        #                vres=self.cfg.VOXEL_SIZE * 2 ** (self.n_scales - i),
        #                dropout=self.cfg.SPARSEREG.DROPOUT)
        #     )
        #     self.tsdf_preds.append(nn.Linear(channels[i], 1))
        #     self.occ_preds.append(nn.Linear(channels[i], 1))

        # grounding_2d
        # self.grounding_2d_model = Grounding_2D()

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
        loss_dict = {}

        # ----generate new coords----
        coords = generate_grid(self.cfg.N_VOX, interval)[0]
        up_coords = []
        for b in range(bs):
            up_coords.append(torch.cat([torch.ones(1, coords.shape[-1]).to(coords.device) * b, coords]))
        up_coords = torch.cat(up_coords, dim=1).permute(1, 0).contiguous()

        # ----gru fusion----
        if self.cfg.FUSION.FUSION_ON:
            up_coords, tsdf_target, occ_target = self.gru_fusion(up_coords, inputs)  # up_coords: 索引
            if self.cfg.FUSION.FULL:  # FULL表示由current和global(history)共同决定
                grid_mask = torch.ones_like(feat[:, 0]).bool()

        # visualization
        # up_coords_numpy = up_coords[:, 1:].detach().cpu().numpy()
        # occ_target_numpy = occ_target.long().detach().cpu().numpy()
        # tsdf_target_numpy = 1 - tsdf_target.abs().detach().cpu().numpy()
        # print('tsdf: ', (tsdf_target_numpy.reshape(-1) > 0).sum())
        # print('occ: ', occ_target_numpy.sum())

        # visualize_mesh(up_coords_numpy[tsdf_target_numpy.reshape(-1) > 0], tsdf_target_numpy[tsdf_target_numpy.reshape(-1) > 0], type='tsdf')
        # visualize_mesh(up_coords_numpy[np.bool_(occ_target_numpy.reshape(-1))], tsdf_target_numpy[np.bool_(occ_target_numpy.reshape(-1))], type='tsdf')
        # # visualize_mesh(up_coords_numpy[(tsdf_target_numpy.reshape(-1) > 0) & (occ_target_numpy.reshape(-1) > 0)],
        # #                tsdf_target_numpy[(tsdf_target_numpy.reshape(-1) > 0) & (occ_target_numpy.reshape(-1) > 0)], type='tsdf')
        # visualize_mesh(up_coords_numpy, occ_target_numpy, type='semantic')

        # -------compute loss-------
        if tsdf_target is not None:
            # loss = self.compute_loss(tsdf, occ, tsdf_target, occ_target,
            #                             mask=grid_mask,
            #                             pos_weight=self.cfg.POS_WEIGHT)
            shape = (24, 24, 24)
            # tsdf2mesh(up_coords[:, 1:] / (2 ** (2 - i)), tsdf, tuple(x * (2 ** i) for x in shape), 'results/tsdf_pred.ply')
            tsdf2mesh(up_coords[:, 1:], tsdf_target, tuple(x * 4 for x in shape), 'results/tsdf_target.ply')
        else:
            loss = torch.Tensor(np.array([0]))[0]
        loss_dict.update({f'tsdf_occ_loss': loss})

        # *********************************************** Grounding_2D ************************************************
        grounding_2d_text = "Find the refrigerator. (with grounding)"
        # 到此为止，仅推理(neural_recon+llava_grounding)，用一张卡，占用显存18G
        grounding_infos = self.grounding_2d_model.inference(img_path=inputs['grounding_img_path'][0], text=grounding_2d_text)  

        ins_class, ins_box, ins_mask = [], [], []
        # 提取所有的object_name
        pattern = r"<g_s>\s*(.*?)\s*<g_e>\s*<seg>"  # 正则表达式匹配, (.*?) 代表非贪婪匹配尽可能少的字符，即提取出中间的object_name, \s* 用来匹配可能出现的空格。
        all_object_names = re.findall(pattern, grounding_infos['response_text'])

        for m in range(len(grounding_infos['response_mask'])):
            for n in range(len(grounding_infos['response_mask'][m])):
                ins_class.append(all_object_names[m])
                ins_box.append(grounding_infos['response_gd'][m][n])
                ins_mask.append(grounding_infos['response_mask'][m][n])  # 256 * 256, 按照向x正方向和y正方向填充成方形区域，对于1296*968，在下方区域填充1/4后resize       

        # 将ins_mask映射到neural_recon的图像中
        for m in range(len(ins_mask)):
           ins_mask[m] = ins_mask[m][:ins_mask[m].shape[0]*3//4, :]
           ins_mask[m] = mask_reshape(input=ins_mask[m], target_size=feats.shape[-2:])

        # 每个voxel对应实例类别 [num_ins, num_voxels]
        ins_volume = []
        for m in range(len(ins_mask)):
            ins_volume.append(ins_back_project(outputs['coords'], inputs['vol_origin_partial'], self.cfg.VOXEL_SIZE, 
                                               ins_mask[m].unsqueeze(0).unsqueeze(0).unsqueeze(0), KRcam, self.cfg.GROUNDING_IMG_IDX)) 
        
        # cv2.imwrite('tmp/2.png', (ins_mask[0].detach().cpu().numpy()*255).astype(np.uint8))
        label2mesh(outputs['coords'][:, 1:], tsdf[occupancy], tuple(x * (2 ** 2) for x in shape), ins_volume[0].float().unsqueeze(1), 'results/label_pred.ply')
        label2mesh(outputs['coords'][:, 1:], tsdf_target[occupancy], tuple(x * (2 ** 2) for x in shape), ins_volume[0].float().unsqueeze(1), 'results/label_target.ply')

        """
        将每个实例对应一个id:
        list 0: (num_voxels), 每个voxel1对应一个id, 0表示背景
        list 1: 
            list 0: dict: {'id': 1, 'category_id': 15}
            ...
        """
        ins_preds_all_batch = []
        for _ in range(bs):
            ins_preds = []
            ins_id_count = 1
            ins_preds.append(torch.zeros_like(outputs['coords'][:, 0]))
            ins_preds.append([])

            for m in range(len(ins_volume)):
                ins_preds[0][ins_volume[m] == True] = ins_id_count
                # ins_preds[1].append({'id': ins_id_count, 'category_id': ins_class[m]})
                ins_preds[1].append({'id': ins_id_count, 'category_id': 15})
                ins_id_count += 1
            
            ins_preds_all_batch.append(ins_preds)
        
        outputs['ins_infos'] = ins_preds_all_batch
        outputs['demand_text'] = grounding_2d_text
        
        return outputs, loss_dict 

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
    tsdf_volume = sparse_to_dense_torch(coords.long(), tsdf, dim_list, 100, tsdf.device, final_tsdf=True)  # 最后的tsdf需要用-1和1填充
    tsdf_volume = tsdf_volume.cpu().numpy()

    verts, faces, norms, vals = measure.marching_cubes(tsdf_volume, level=0)  # level: 等值面的标量值，函数将提取此标量值对应的表面
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms)
    mesh.export(save_path)

def label2mesh(coords, tsdf, dim_list, label, save_path):
    tsdf = tsdf.view(-1).detach()
    tsdf_volume = sparse_to_dense_torch(coords.long(), tsdf, dim_list, 100, tsdf.device, final_tsdf=True)  # 最后的tsdf需要用-1和1填充
    tsdf_volume = tsdf_volume.cpu().numpy()

    label = label.view(-1).detach()
    label_volume = sparse_to_dense_torch(coords.long(), label, dim_list, 0, label.device, final_tsdf=False)  # 0填充
    label_volume = label_volume.cpu().numpy()

    # Step 1: 提取mesh
    verts, faces, norms, vals = measure.marching_cubes(tsdf_volume, level=0)  # level: 等值面的标量值，函数将提取此标量值对应的表面
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms)

    # Step 2: 将label映射到mesh的顶点
    # 使用最近邻插值，找到每个顶点在 label_volume 中对应的坐标，并获取标签
    verts_scaled = (verts / np.array(tsdf_volume.shape)) * np.array(label_volume.shape)  # 缩放顶点到label_volume的范围
    verts_indices = np.round(verts_scaled).astype(int)  # 取最近邻的整数索引
    verts_indices = np.clip(verts_indices, 0, np.array(label_volume.shape) - 1)  # 防止越界
    vertex_labels = label_volume[verts_indices[:, 0], verts_indices[:, 1], verts_indices[:, 2]]

    # Step 3: 给每个标签分配颜色
    unique_labels = np.unique(vertex_labels)
    # 使用colormap为不同的label生成颜色
    colormap = cm.get_cmap('tab20', len(unique_labels))  # 可以选择其他的colormap，比如 'jet'， 'viridis' 等
    # 定义标签到颜色的映射，类别0强制为黑色，其余类别使用colormap
    label_to_color = {}
    for i, label in enumerate(unique_labels):
        if label == 0:
            label_to_color[label] = np.array([192, 192, 192], dtype=np.uint8)  # 类别0为黑色
        else:
            label_to_color[label] = (np.array(colormap(i)[:3]) * 255).astype(np.uint8)
    
    
    # 为每个顶点分配颜色
    vertex_colors = np.array([label_to_color[label] for label in vertex_labels])

    # Step 4: 保存到PLY文件，包括顶点、面片、法线和颜色
    mesh.visual.vertex_colors = vertex_colors
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