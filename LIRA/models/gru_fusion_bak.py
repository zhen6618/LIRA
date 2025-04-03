import torch
import torch.nn as nn
from torchsparse.tensor import PointTensor
from neu_utils import sparse_to_dense_channel, sparse_to_dense_torch
from .modules import ConvGRU


class GRUFusion(nn.Module):
    """
    Two functionalities of this class:
    1. GRU Fusion module as in the paper. Update hidden state features with ConvGRU.
    2. Substitute TSDF in the global volume when direct_substitute = True.
    """

    def __init__(self, cfg, ch_in=None, direct_substitute=False):
        super(GRUFusion, self).__init__()
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
        self.global_volume = [None, None, None]
        self.target_tsdf_volume = [None, None, None]

        self.global_instance = [None]  # instance_id
        self.global_semantic = [None]  # semantic_id

        if direct_substitute:
            self.fusion_nets = None
        else:
            self.fusion_nets = nn.ModuleList()
            for i, ch in enumerate(ch_in):
                self.fusion_nets.append(ConvGRU(hidden_dim=ch,
                                                input_dim=ch,
                                                pres=1,
                                                vres=self.cfg.VOXEL_SIZE * 2 ** (self.n_scales - i)))

    def reset(self, i):
        self.global_volume[i] = PointTensor(torch.Tensor([]), torch.Tensor([]).view(0, 3).long()).cuda()
        self.target_tsdf_volume[i] = PointTensor(torch.Tensor([]), torch.Tensor([]).view(0, 3).long()).cuda()

        self.global_instance = torch.empty(0).cuda()  # instance_id
        self.global_semantic = torch.empty(0).cuda()  # semantic_id

    def convert2dense(self, current_coords, current_values, coords_target_global, tsdf_target, relative_origin,
                      scale):
        '''
        1. convert sparse feature to dense feature;
        2. combine current feature coordinates and previous coordinates within FBV from global hidden state to get
        new feature coordinates (updated_coords);
        3. fuse ground truth tsdf.

        :param current_coords: (Tensor), current coordinates, (N, 3)
        :param current_values: (Tensor), current features/tsdf, (N, C)
        :param coords_target_global: (Tensor), ground truth coordinates, (N', 3)
        :param tsdf_target: (Tensor), tsdf ground truth, (N',)
        :param relative_origin: (Tensor), origin in global volume, (3,)
        :param scale:
        :return: updated_coords: (Tensor), coordinates after combination, (N', 3)
        :return: current_volume: (Tensor), current dense feature/tsdf volume, (DIM_X, DIM_Y, DIM_Z, C)
        :return: global_volume: (Tensor), global dense feature/tsdf volume, (DIM_X, DIM_Y, DIM_Z, C)
        :return: target_volume: (Tensor), dense target tsdf volume, (DIM_X, DIM_Y, DIM_Z, 1)
        :return: valid: mask: 1 represent in current FBV (N,)
        :return: valid_target: gt mask: 1 represent in current FBV (N,)
        '''
        # previous frame
        global_coords = self.global_volume[scale].C
        global_value = self.global_volume[scale].F
        global_tsdf_target = self.target_tsdf_volume[scale].F
        global_coords_target = self.target_tsdf_volume[scale].C

        # dim = (torch.Tensor(self.cfg.N_VOX).cuda() // 2 ** (self.cfg.N_LAYER - scale - 1)).int()
        dim = (torch.div(torch.Tensor(self.cfg.N_VOX).cuda(), 2 ** (self.cfg.N_LAYER - scale - 1), rounding_mode='floor')).int()
        dim_list = dim.data.cpu().numpy().tolist()

        # mask voxels that are out of the FBV
        global_coords = global_coords - relative_origin
        valid = ((global_coords < dim) & (global_coords >= 0)).all(dim=-1)
        if self.cfg.FUSION.FULL is False:  # 只取current coords内， 相当于把current coords内的current feats和global feats融合到一起
            valid_volume = sparse_to_dense_torch(current_coords, 1, dim_list, 0, global_value.device)
            value = valid_volume[global_coords[valid][:, 0], global_coords[valid][:, 1], global_coords[valid][:, 2]]
            all_true = valid[valid]
            all_true[value == 0] = False
            valid[valid] = all_true
        # sparse to dense  补全整个volume变成dense，再进行coordinates拼接
        global_volume = sparse_to_dense_channel(global_coords[valid], global_value[valid], dim_list, self.ch_in[scale],
                                                self.feat_init, global_value.device)  # 特征

        current_volume = sparse_to_dense_channel(current_coords, current_values, dim_list, self.ch_in[scale],
                                                 self.feat_init, global_value.device)  # 特征

        if self.cfg.FUSION.FULL is True:
            # change the structure of sparsity, combine current coordinates and previous coordinates from global volume
            if self.direct_substitude:
                updated_coords = torch.nonzero((global_volume.abs() < 1).any(-1) | (current_volume.abs() < 1).any(-1))
            else:
                updated_coords = torch.nonzero((global_volume != 0).any(-1) | (current_volume != 0).any(-1))
        else:
            updated_coords = current_coords

        # fuse ground truth
        if tsdf_target is not None:
            # mask voxels that are out of the FBV
            global_coords_target = global_coords_target - relative_origin
            valid_target = ((global_coords_target < dim) & (global_coords_target >= 0)).all(dim=-1)
            # combine current tsdf and global tsdf
            coords_target = torch.cat([global_coords_target[valid_target], coords_target_global])[:, :3]
            tsdf_target = torch.cat([global_tsdf_target[valid_target], tsdf_target.unsqueeze(-1)])
            # sparse to dense
            target_volume = sparse_to_dense_channel(coords_target, tsdf_target, dim_list, 1, 1,
                                                    tsdf_target.device)
        else:
            target_volume = valid_target = None

        return updated_coords, current_volume, global_volume, target_volume, valid, valid_target

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

    def ins_fusion(self, scale, global_valid, relative_origin, ins_info, current_coords):

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
        history_matrix = (self.global_volume[scale].C[global_valid].unsqueeze(1) == current_coords.unsqueeze(0)).all(dim=-1)
        if history_matrix.shape[0] != 0:
            matched_current_indices = history_matrix.any(dim=0)  # 找到global中每个点对应current中的匹配索引
            matched_global_indices = history_matrix.float().argmax(dim=0)  # 获取global中的匹配索引

            new_current_instance[matched_current_indices] = global_instance[matched_global_indices[matched_current_indices]].reshape(-1).to(new_current_instance.dtype)
            new_current_semantic[matched_current_indices] = global_semantic[matched_global_indices[matched_current_indices]].reshape(-1).to(new_current_semantic.dtype)

        # current更新
        num_current_instance = len(ins_info[1])
        increment_count = 1  # 用于新增instance_id数
        for i in range(num_current_instance):
            cls = ins_info[1][i]['category_id']

            if cls in global_semantic:
                cls_global_index = global_semantic == cls  # in current volume
                ins_global = global_instance[cls_global_index]
                ins_global_all = torch.unique(ins_global)

                match_flag = False
                # 取出global中所有是这个instance_id的点
                for ins_id in ins_global_all:
                    comparison_global_index = self.global_instance == ins_id  # in global volume
                    comparison_global_coords = self.global_volume[scale].C[comparison_global_index.view(-1)]

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

    def update_map(self, value, coords, target_volume, valid, valid_target,
                   relative_origin, scale, new_current_instance=None, new_current_semantic=None):
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
        # pred
        self.global_volume[scale].F = torch.cat([self.global_volume[scale].F[valid == False], value])
        coords = coords + relative_origin
        self.global_volume[scale].C = torch.cat([self.global_volume[scale].C[valid == False], coords])

        if self.direct_substitude:
            self.global_instance = torch.cat([self.global_instance[valid == False], new_current_instance])  # 直接替换历史实例
            self.global_semantic = torch.cat([self.global_semantic[valid == False], new_current_semantic])

        # target
        if target_volume is not None:
            target_volume = target_volume.squeeze()
            self.target_tsdf_volume[scale].F = torch.cat(
                [self.target_tsdf_volume[scale].F[valid_target == False],
                 target_volume[target_volume.abs() < 1].unsqueeze(-1)])
            target_coords = torch.nonzero(target_volume.abs() < 1) + relative_origin

            self.target_tsdf_volume[scale].C = torch.cat(
                [self.target_tsdf_volume[scale].C[valid_target == False], target_coords])

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

    def forward(self, coords, values_in, inputs, scale=2, outputs=None, save_mesh=False, ins_infos=None):
        if self.global_volume[scale] is not None:
            # delete computational graph to save memory
            self.global_volume[scale] = self.global_volume[scale].detach()

        batch_size = len(inputs['fragment'])
        interval = 2 ** (self.cfg.N_LAYER - scale - 1)  # self.cfg.N_LAYER: 模型总共有3层(粗到精)；scale：当前在哪一层

        tsdf_target_all = None
        occ_target_all = None
        values_all = None
        updated_coords_all = None

        # ---incremental fusion----
        for i in range(batch_size):
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

            batch_ind = torch.nonzero(coords[:, 0] == i).squeeze(1)
            if len(batch_ind) == 0:
                continue
            # coords_b = coords[batch_ind, 1:].long() // interval
            coords_b = torch.div(coords[batch_ind, 1:].long(), interval, rounding_mode='floor')  # [N_voxels, 3]
            values = values_in[batch_ind]

            # 获取gt
            if 'occ_list' in inputs.keys():
                # get partial gt
                occ_target = inputs['occ_list'][self.cfg.N_LAYER - scale - 1][i]
                tsdf_target = inputs['tsdf_list'][self.cfg.N_LAYER - scale - 1][i][occ_target]
                coords_target = torch.nonzero(occ_target)
            else:
                coords_target = tsdf_target = None

            # convert to dense: 1. convert sparse feature to dense feature; 2. combine current feature coordinates and
            # previous feature coordinates within FBV from our backend map to get new feature coordinates (updated_coords)
            updated_coords, current_volume, global_volume, target_volume, valid, valid_target = self.convert2dense(
                coords_b,
                values,  # values就是feats
                coords_target,
                tsdf_target,
                relative_origin,
                scale)

            # dense to sparse: get features using new feature coordinates (updated_coords)  sparse: (x_dim*y_dim*z_dim, ), dense: (x_dim, y_dim, z_dim, c)
            values = current_volume[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]]  # 当前特征 (voxel_num, c)
            global_values = global_volume[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]]  # 历史全局特征 (voxel_num, c)

            # get fused gt
            if target_volume is not None:
                tsdf_target = target_volume[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]]  # (voxel_num, 1)
                occ_target = tsdf_target.abs() < 1  # (voxel_num, 1)
            else:
                tsdf_target = occ_target = None

            if not self.direct_substitude:  # 不直接替代，即进行GRUFusion
                # convert to aligned camera coordinate (转换到Fragment坐标系)
                r_coords = updated_coords.detach().clone().float()
                r_coords = r_coords.permute(1, 0).contiguous().float() * voxel_size + origin.unsqueeze(-1).float()
                r_coords = torch.cat((r_coords, torch.ones_like(r_coords[:1])), dim=0)
                r_coords = inputs['world_to_aligned_camera'][i, :3, :] @ r_coords
                r_coords = torch.cat([r_coords, torch.zeros(1, r_coords.shape[-1]).to(r_coords.device)])
                r_coords = r_coords.permute(1, 0).contiguous()

                # ConvGRU
                h = PointTensor(global_values, r_coords)  # global_values:[N_voxels, c], r_coords:[N_voxels, 4]
                x = PointTensor(values, r_coords)  # values:[N_voxels, c], r_coords:[N_voxels, 4]
                values = self.fusion_nets[scale](h, x)

            if self.direct_substitude:
                dim = (torch.div(torch.Tensor(self.cfg.N_VOX).cuda(), 2 ** (self.cfg.N_LAYER - scale - 1), rounding_mode='floor')).int()
                dim_list = dim.data.cpu().numpy().tolist()
                current_instance_volume = sparse_to_dense_channel(coords_b, ins_infos[i][0].unsqueeze(1), dim_list, 1, 0, values.device)
                ins_infos[i][0] = current_instance_volume[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]].squeeze(1)

                # new_current_instance: [N, ], new_current_semantic: [N, ]
                new_current_instance, new_current_semantic = self.ins_fusion(scale=scale,
                                                                             global_valid=valid,
                                                                             relative_origin=relative_origin,
                                                                             ins_info=ins_infos[i],
                                                                             current_coords=updated_coords)

            # feed back to global volume (direct substitute)  fragment融合， global直接替代
            if not self.direct_substitude:
                self.update_map(values, updated_coords, target_volume, valid, valid_target, relative_origin, scale)
            else:
                self.update_map(values, updated_coords, target_volume, valid, valid_target, relative_origin, scale, new_current_instance.view(-1, 1), new_current_semantic.view(-1, 1))

            if updated_coords_all is None:
                updated_coords_all = torch.cat([torch.ones_like(updated_coords[:, :1]) * i, updated_coords * interval],
                                               dim=1)
                values_all = values
                tsdf_target_all = tsdf_target
                occ_target_all = occ_target
            else:
                updated_coords = torch.cat([torch.ones_like(updated_coords[:, :1]) * i, updated_coords * interval],
                                           dim=1)
                updated_coords_all = torch.cat([updated_coords_all, updated_coords])
                values_all = torch.cat([values_all, values])
                if tsdf_target_all is not None:
                    tsdf_target_all = torch.cat([tsdf_target_all, tsdf_target])
                    occ_target_all = torch.cat([occ_target_all, occ_target])

            if self.direct_substitude and save_mesh:
                outputs = self.save_mesh(scale, outputs, self.scene_name[scale])

        if self.direct_substitude:
            return outputs
        else:
            return updated_coords_all, values_all, tsdf_target_all, occ_target_all
