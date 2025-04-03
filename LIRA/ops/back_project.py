import torch
from torch.nn.functional import grid_sample
import torch.nn.functional as F
import open3d as o3d
import numpy as np


def back_project(coords, origin, voxel_size, feats, KRcam):
    '''
    Unproject the image fetures to form a 3D (sparse) feature volume

    :param coords: coordinates of voxels,
    dim: (num of voxels, 4) (4 : batch ind, x, y, z)
    :param origin: origin of the partial voxel volume (xyz position of voxel (0, 0, 0))
    dim: (batch size, 3) (3: x, y, z)
    :param voxel_size: floats specifying the size of a voxel
    :param feats: image features
    dim: (num of views, batch size, C, H, W)
    :param KRcam: projection matrix
    dim: (num of views, batch size, 4, 4)
    :return: feature_volume_all: 3D feature volumes
    dim: (num of voxels, c + 1)
    :return: count: number of times each voxel can be seen
    dim: (num of voxels,)
    '''
    n_views, bs, c, h, w = feats.shape

    feature_volume_all = torch.zeros(coords.shape[0], c + 1).cuda()  # +1是存储metadata, 为voxel在相机坐标系下的深度值
    count = torch.zeros(coords.shape[0]).cuda()

    for batch in range(bs):
        batch_ind = torch.nonzero(coords[:, 0] == batch).squeeze(1)
        coords_batch = coords[batch_ind][:, 1:]  # coords: [num_voxel, bxyz]

        coords_batch = coords_batch.view(-1, 3)
        origin_batch = origin[batch].unsqueeze(0)
        feats_batch = feats[:, batch]
        proj_batch = KRcam[:, batch]

        grid_batch = coords_batch * voxel_size + origin_batch.float()  # 转换成实际坐标 origin_batch: 起始位置, grid_batch: 目标位置
        rs_grid = grid_batch.unsqueeze(0).expand(n_views, -1, -1)
        rs_grid = rs_grid.permute(0, 2, 1).contiguous()
        nV = rs_grid.shape[-1]  # voxel总数量
        rs_grid = torch.cat([rs_grid, torch.ones([n_views, 1, nV]).cuda()], dim=1)  # 齐次坐标形式

        # Project grid
        im_p = proj_batch @ rs_grid  # 每个voxel的实际坐标映射到图像坐标系
        im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]
        im_x = im_x / im_z  # 获取每个view的像素坐标
        im_y = im_y / im_z

        # voxel投影之后在图像视野之外的mask掉
        im_grid = torch.stack([2 * im_x / (w - 1) - 1, 2 * im_y / (h - 1) - 1], dim=-1)
        mask = im_grid.abs() <= 1
        mask = (mask.sum(dim=-1) == 2) & (im_z > 0)

        feats_batch = feats_batch.view(n_views, c, h, w)
        im_grid = im_grid.view(n_views, 1, -1, 2)
        features = grid_sample(feats_batch, im_grid, padding_mode='zeros', align_corners=True)  # 为每个voxel取对应的img feats, 插值采样  # [N_views, dim, bs, N_voxels]

        # 过滤, 去除nan值等
        features = features.view(n_views, c, -1)
        mask = mask.view(n_views, -1)
        im_z = im_z.view(n_views, -1)
        # remove nan, 令voxel对应位置为0
        features[mask.unsqueeze(1).expand(-1, c, -1) == False] = 0
        im_z[mask == False] = 0

        count[batch_ind] = mask.sum(dim=0).float()

        # aggregate multi view
        features = features.sum(dim=0)  # 9个view的feats取平均
        mask = mask.sum(dim=0)
        invalid_mask = mask == 0
        mask[invalid_mask] = 1  # 防止对图像特征平均的时候➗0，改成➗1
        in_scope_mask = mask.unsqueeze(0)
        features /= in_scope_mask  # 取平均
        features = features.permute(1, 0).contiguous()  # [N_voxels, dim]

        # concat normalized depth value
        im_z = im_z.sum(dim=0).unsqueeze(1) / in_scope_mask.permute(1, 0).contiguous()
        im_z_mean = im_z[im_z > 0].mean()
        im_z_std = torch.norm(im_z[im_z > 0] - im_z_mean) + 1e-5
        im_z_norm = (im_z - im_z_mean) / im_z_std
        im_z_norm[im_z <= 0] = 0
        features = torch.cat([features, im_z_norm], dim=1)  # feats特征 concat 标准化之后的深度信息

        feature_volume_all[batch_ind] = features
    return feature_volume_all, count


def mask_reshape(input, target_size):

    float_tensor = input.float()
    resized_tensor = F.interpolate(float_tensor.unsqueeze(0).unsqueeze(0), size=(target_size[0], target_size[1]), mode='nearest')
    resized_bool_tensor = resized_tensor.squeeze(0).squeeze(0).bool()

    return resized_bool_tensor


def ins_back_project(coords, origin, voxel_size, feats, KRcam, grounding_idx):
    n_views, bs, c, h, w = feats.shape

    KRcam = KRcam[grounding_idx, ...].unsqueeze(0)

    feature_volume_all = torch.zeros(coords.shape[0], c).bool().cuda()  # +1是存储metadata, 为voxel在相机坐标系下的深度值

    for batch in range(bs):
        batch_ind = torch.nonzero(coords[:, 0] == batch).squeeze(1)
        coords_batch = coords[batch_ind][:, 1:]  # coords: [num_voxel, bxyz]

        coords_batch = coords_batch.view(-1, 3)
        origin_batch = origin[batch].unsqueeze(0)
        feats_batch = feats[:, batch]
        proj_batch = KRcam[:, batch]

        grid_batch = coords_batch * voxel_size + origin_batch.float()  # 转换成实际坐标 origin_batch: 起始位置, grid_batch: 目标位置
        rs_grid = grid_batch.unsqueeze(0).expand(n_views, -1, -1)
        rs_grid = rs_grid.permute(0, 2, 1).contiguous()
        nV = rs_grid.shape[-1]  # voxel总数量
        rs_grid = torch.cat([rs_grid, torch.ones([n_views, 1, nV]).cuda()], dim=1)  # 齐次坐标形式

        # Project grid
        im_p = proj_batch @ rs_grid  # 每个voxel的实际坐标映射到图像坐标系
        im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]
        im_x = im_x / im_z  # 获取每个view的像素坐标
        im_y = im_y / im_z

        # voxel投影之后在图像视野之外的mask掉
        im_grid = torch.stack([2 * im_x / (w - 1) - 1, 2 * im_y / (h - 1) - 1], dim=-1)
        mask = im_grid.abs() <= 1
        mask = (mask.sum(dim=-1) == 2) & (im_z > 0)

        feats_batch = feats_batch.view(n_views, c, h, w)
        im_grid = im_grid.view(n_views, 1, -1, 2)
        feats_batch = feats_batch.float()
        features = grid_sample(feats_batch, im_grid, padding_mode='zeros', align_corners=True)  # 为每个voxel取对应的img feats, 插值采样  # [N_views, dim, bs, N_voxels]

        # 过滤, 去除nan值等
        features = features.view(n_views, c, -1)
        mask = mask.view(n_views, -1)
        # remove nan, 令voxel对应位置为0
        features[mask.unsqueeze(1).expand(-1, c, -1) == False] = 0

        features = features.bool()
        features = features.squeeze(0)
        features = features.permute(1, 0).contiguous()

        feature_volume_all[batch_ind] = features

    return feature_volume_all.squeeze(1)


def rs_ins_back_project(coords_batch, origin_batch, voxel_size, feats_batch, KRcam, grounding_idx, depth, is_bool_tensor=False):
    c, h, w = feats_batch.shape
    feats_batch = feats_batch.permute(1, 2, 0)

    proj_batch = KRcam[grounding_idx]
    origin_batch = origin_batch.unsqueeze(0)
    coords_batch = coords_batch.view(-1, 3)

    grid_batch = coords_batch * voxel_size + origin_batch.float()  # 转换成实际坐标 origin_batch: 起始位置, grid_batch: 目标位置
    rs_grid = grid_batch
    rs_grid = rs_grid.permute(1, 0).contiguous()
    nV = rs_grid.shape[-1]  # voxel总数量
    rs_grid = torch.cat([rs_grid, torch.ones([1, nV]).cuda()], dim=0)  # 齐次坐标形式

    # Project grid
    im_p = proj_batch @ rs_grid  # 每个voxel的实际坐标映射到图像坐标系
    im_x, im_y, im_z = im_p[0], im_p[1], im_p[2]
    im_x = im_x / im_z  # 获取每个view的像素坐标
    im_y = im_y / im_z

    # voxel投影之后在图像视野之外的mask掉
    im_grid = torch.stack([2 * im_x / (w - 1) - 1, 2 * im_y / (h - 1) - 1], dim=-1)
    mask = im_grid.abs() <= 1
    mask = (mask.sum(dim=-1) == 2) & (im_z > 0)

    # pixels = torch.stack([im_x, im_y], dim=-1)  # [N_voxels, 2]
    # pixel_coords = pixels.round().long()
    
    # valid_voxels = coords_batch[mask]
    valid_cam_coords = im_p.permute(1, 0)[mask]
    # valid_pixel_coords = pixel_coords[mask]

    '***********************   最近点映射   ***********************'
    valid_cam_coords = rs_grid.permute(1, 0)[mask][:, :3]

    valid_depth_mask = depth > 0
    valid_depths = depth[valid_depth_mask]  # [n]
    valid_feats = feats_batch[valid_depth_mask]  # [n, c]

    valid_rows, valid_cols = torch.nonzero(valid_depth_mask, as_tuple=True)
    pixel_coordinates = torch.stack([valid_cols*valid_depths, valid_rows*valid_depths, valid_depths], dim=1)
    pixel_coordinates_homogeneous = torch.cat([pixel_coordinates, torch.ones_like(pixel_coordinates[:, :1])], dim=1)

    P_inv = torch.inverse(proj_batch)
    world_coordinates_homogeneous = torch.matmul(P_inv, pixel_coordinates_homogeneous.unsqueeze(-1)).squeeze(-1)
    world_coordinates = world_coordinates_homogeneous[:, :3] / world_coordinates_homogeneous[:, 3:].expand_as(world_coordinates_homogeneous[:, :3])

    # world_coordinates_numpy = world_coordinates.detach().cpu().numpy()
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(world_coordinates_numpy)
    # o3d.io.write_point_cloud("depth.ply", point_cloud)

    distances = torch.cdist(world_coordinates, valid_cam_coords)
    min_distances, min_indices = torch.min(distances, dim=0)
    valid_min_mask = (min_distances < voxel_size)  # 对应点距离小于voxel_size
    valid_min_indices = min_indices[valid_min_mask]  # 对应的最小距离索引

    dis_min_feats = torch.zeros((valid_cam_coords.shape[0], c), device=feats_batch.device, dtype=feats_batch.dtype)
    dis_min_feats[valid_min_mask] = valid_feats[valid_min_indices]

    assigned_features = torch.zeros((nV, c), device=feats_batch.device, dtype=feats_batch.dtype)
    assigned_features[mask] = dis_min_feats

    # vis_depth(coords_batch.detach().cpu().numpy(), assigned_features.reshape(-1).float().detach().cpu().numpy(), 'results/label_target_depth.ply')

    return assigned_features.squeeze(1)

    # '***********************   处理深度值   ***********************'
    # valid_depth_mask = depth > 0  # invaild深度值为0
    # valid_depths = depth[valid_depth_mask]
    # valid_rows, valid_cols = torch.nonzero(valid_depth_mask, as_tuple=True)  # 转置

    # pixel_coordinates = torch.stack([valid_cols*valid_depths, valid_rows*valid_depths, valid_depths], dim=1)
    # pixel_coordinates_homogeneous = torch.cat([pixel_coordinates, torch.ones_like(pixel_coordinates[:, :1])], dim=1)
    # "计算投影矩阵的逆（注意：这可能在数值上不稳定）"
    # P_inv = torch.inverse(proj_batch)
    # world_coordinates_homogeneous = torch.matmul(P_inv, pixel_coordinates_homogeneous.unsqueeze(-1)).squeeze(-1)
    # world_coordinates = world_coordinates_homogeneous[:, :3] / world_coordinates_homogeneous[:, 3:].expand_as(world_coordinates_homogeneous[:, :3])

    # world_coordinates_numpy = world_coordinates.detach().cpu().numpy()
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(world_coordinates_numpy)
    # o3d.io.write_point_cloud("depth.ply", point_cloud)


    # '***********************   只投影到射线上距离相机最近的体素   ***********************'
    # # 按射线深度对 voxel 排序，确保最近的在前
    # '计算外参矩阵'
    # # K = torch.tensor([[1169.621094, 0.000000, 646.295044],
    # #               [0.000000, 1167.105103, 489.927032],
    # #               [0.000000, 0.000000, 1.000000]], device=coords_batch.device)
    # # # 计算内参矩阵的逆
    # # K_inv = torch.linalg.inv(K)
    # # # 提取投影矩阵的前三列 (3x3)
    # # proj_matrix_3x3 = proj_batch[:3, :3]  # 取前3行和前3列
    # # # 计算旋转矩阵 R
    # # R = torch.matmul(K_inv, proj_matrix_3x3)
    # # # 提取投影矩阵的第四列 (平移向量)
    # # t = proj_batch[:3, 3]
    # # # 合并旋转矩阵 R 和位移向量 t 组成外参矩阵 [R | t]
    # # extrinsics = torch.cat([R, t.unsqueeze(1)], dim=1)
    # # P_camera = torch.matmul(extrinsics, rs_grid)
    # # depths = P_camera.permute(1, 0)[mask][:, 2]

    # depths = valid_cam_coords[:, 2]
    # sorted_indices = torch.argsort(depths, descending=False)  # descending=False: 从到大排序
    # valid_voxels = valid_voxels[sorted_indices]
    # valid_pixel_coords = valid_pixel_coords[sorted_indices]
    # depths = depths[sorted_indices]

    # # # Initialize the feature map container
    # # assigned_features = torch.zeros((nV, c), device=coords_batch.device, dtype=feats_batch.dtype)
    
    # # # Compute unique pixel coordinates
    # # unique_pixel_coords = valid_pixel_coords.unique(dim=0, return_inverse=True)

    # # # Use scatter to assign the features to the correct voxel positions
    # # pixel_indices = unique_pixel_coords[1]
    # # assigned_features.scatter_(0, pixel_indices.unsqueeze(1).expand(-1, feats_batch.shape[0]), feats_batch[:, valid_pixel_coords[:, 1], valid_pixel_coords[:, 0]].T)


    # # # 找到每条射线（像素）上的最近 voxel
    # # unique_pixel_coords, first_indices = torch.unique(valid_pixel_coords, return_inverse=True, dim=0)
    # # nearest_indices = sorted_indices[first_indices]  # 最近的 voxel 索引

    # # # 从特征图中提取最近像素的特征
    # # src_features = feats_batch[:, unique_pixel_coords[:, 1], unique_pixel_coords[:, 0]].T  # 转置为 (n_unique, C)

    # # # 初始化目标特征张量并分配特征
    # # assigned_features = torch.zeros((nV, c), device=coords_batch.device, dtype=feats_batch.dtype)
    # # assigned_features[nearest_indices] = src_features  # Shape: (N, C)

    # # 找到每条射线（像素）上的最近 voxel
    # unique_pixel_coords, inverse_indices = torch.unique(valid_pixel_coords, return_inverse=True, dim=0)
    # nearest_indices = torch.zeros(unique_pixel_coords.size(0), device=coords_batch.device, dtype=torch.long)  

    # nes_in = torch.zeros((unique_pixel_coords.shape[0]), device=coords_batch.device, dtype=unique_pixel_coords.dtype)
    # for idx in range(unique_pixel_coords.shape[0]):
    #     nes_find = torch.where(inverse_indices == idx)[0][0].item()
    #     nes_in[idx] = nes_find
    
    # # 使用scatter方法记录每个元素第一次出现的位置, scatter函数后面重复出现的会覆盖以前的值
    # nearest_indices = torch.scatter(nearest_indices, 0, inverse_indices, torch.arange(valid_pixel_coords.size(0), device=coords_batch.device))

    # # 从特征图中提取最近像素的特征
    # src_features = feats_batch[:, unique_pixel_coords[:, 1], unique_pixel_coords[:, 0]].T  # 转置为 (n_unique, C)
    # # src_features = feats_batch[:, valid_pixel_coords[:, 1], valid_pixel_coords[:, 0]].T

    # # 初始化目标特征张量并分配特征
    # assigned_features = torch.zeros((nV, c), device=coords_batch.device, dtype=feats_batch.dtype)

    # assigned_features.scatter_(0, mask.nonzero()[sorted_indices][nearest_indices].expand(-1, c), src_features)  # Shape: (N, C) 
    # # assigned_features.scatter_(0, mask.nonzero()[sorted_indices].expand(-1, c), src_features)

    # normalized_depths = (depths - depths.min()) / (depths.max() - depths.min())
    # vis_depths = torch.zeros((nV), device=coords_batch.device, dtype=depths.dtype)
    # vis_depths.scatter_(0, mask.nonzero()[:, 0][sorted_indices], normalized_depths)

    # vis_depth(coords_batch.detach().cpu().numpy(), vis_depths.detach().cpu().numpy(), 'results/label_target.ply')

    # # return assigned_features.squeeze(1)
    # return vis_depths


    # # feats_batch = feats_batch.view(n_views, c, h, w)
    # # im_grid = im_grid.view(n_views, 1, -1, 2)
    # # feats_batch = feats_batch.float()

    # # '***********************   只投影到射线上距离相机最近的体素   ***********************'
    # # # 计算每个像素到所有体素的距离（这里只考虑像素-体素投影射线的距离）
    # # voxel_positions = rs_grid[:, 0:3, :]  # 体素位置 [n_views, 3, n_voxels]
    
    # # # 扩展 pixel_positions 以便计算 3D 距离
    # # pixel_positions = torch.stack([im_x, im_y, im_z], dim=-1)  # [n_views, n_voxels, 3]

    # # # 计算像素与体素之间的欧几里得距离
    # # # 使用 broadcast 使得两者的维度匹配：pixel_positions (n_views, n_voxels, 3) 和 voxel_positions (n_views, 3, n_voxels)
    # # distances = torch.norm(voxel_positions - pixel_positions.permute(0, 2, 1), dim=1)  # [n_views, n_voxels]

    # # # 找到每个像素距离最小的体素
    # # min_distances, nearest_voxel_idx = torch.min(distances, dim=-1)  # [n_views, 1] 返回距离最近的体素索引

    # # # 通过索引过滤，选择每个像素最近的体素的特征
    # # nearest_voxel_feats = feats_batch.view(n_views, c, -1)[:, :, nearest_voxel_idx]

    # # # 将采样得到的特征与mask结合，去除无效点
    # # nearest_voxel_feats = nearest_voxel_feats.view(n_views, c, -1)
    # # mask = mask.view(n_views, -1)
    # # nearest_voxel_feats[mask.unsqueeze(1).expand(-1, c, -1) == False] = 0

    # # if is_bool_tensor:
    # #     nearest_voxel_feats = nearest_voxel_feats.bool()

    # # nearest_voxel_feats = nearest_voxel_feats.squeeze(0)
    # # nearest_voxel_feats = nearest_voxel_feats.permute(1, 0).contiguous()

    # # feature_volume_all = nearest_voxel_feats

    # # # features = grid_sample(feats_batch, im_grid, padding_mode='zeros', align_corners=True)  # 为每个voxel取对应的img feats, 插值采样  # [N_views, dim, bs, N_voxels]

    # # # # 过滤, 去除nan值等
    # # # features = features.view(n_views, c, -1)
    # # # mask = mask.view(n_views, -1)
    # # # # remove nan, 令voxel对应位置为0
    # # # features[mask.unsqueeze(1).expand(-1, c, -1) == False] = 0

    # # # if is_bool_tensor:
    # # #     features = features.bool()
    # # # features = features.squeeze(0)
    # # # features = features.permute(1, 0).contiguous()

    # # # feature_volume_all = features

    # # return feature_volume_all.squeeze(1)

def vis_depth(voxel_coords, heat_values, filename):          
# 将热力值归一化到 [0, 255] 范围内并作为 RGB 颜色
    # heat_values = ins_volume[0].detach().cpu().numpy()
    # heat_values = heat_values[heat_values != 0]
    # voxel_coords = up_coords_batch[:, 1:].detach().cpu().numpy()
    # voxel_coords = voxel_coords[heat_values != 0]
    heat_values_normalized = (heat_values * 255).astype(np.uint8)  # 将热力值缩放到 0-255 范围
    colors = np.tile(heat_values_normalized[:, np.newaxis], (1, 3))  # 转换为 RGB
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_coords)  # 设置体素坐标
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # 设置颜色，必须是 [0, 1] 范围
    o3d.io.write_point_cloud(filename, pcd)
