import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from mpl_toolkits.mplot3d import Axes3D
import torch
from skimage import measure
import trimesh
from matplotlib import cm
from scipy.spatial import Delaunay


def plot_voxels(voxels, title="Voxel Grid"):
    """ 使用pyvista可视化3D体素数据 """
    # 创建一个空的网格
    grid = pv.UniformGrid()

    # 设置网格的尺寸：加1因为点数比单元数多1
    grid.dimensions = np.array(voxels.shape) + 1

    # 设置网格的原点
    grid.origin = (0, 0, 0)  # The bottom left corner of the data set

    # 设置网格的间距
    grid.spacing = (1, 1, 1)  # These are the cell sizes along each axis

    # 将体素数据添加到网格中
    grid.cell_data["values"] = voxels.flatten(order="F")  # Flatten the array!

    # 创建一个绘图对象并添加网格
    plotter = pv.Plotter()
    plotter.add_volume(grid, scalars="values", opacity='sigmoid', show_scalar_bar=False)
    plotter.set_background("white")
    plotter.camera_position = 'iso'
    plotter.add_title(title)
    plotter.show()


def visualize_mesh(input_A=None, input_B=None, type='xyz', use_path=False, out_file=None):

    # A = np.load('datasets/scannet/panoptic_info/scene0000_00_vert.npy')  xyz(N, 3) or xyzrgb(N, 6)
    # B = np.load('datasets/scannet/panoptic_info/scene0000_00_sem_label.npy')  class_id(N,)
    # C = np.load('datasets/scannet/panoptic_info/scene0000_00_ins_label.npy')  instance_id(N,)

    if use_path:
        A = np.load(input_A)
    else:
        A = input_A
    N = A.shape[0]
    A = A.reshape(N, -1)
    if type == 'xyz':
        # 创建点云数据
        points = pv.PolyData(A[:, :3])

        # 创建灰色的RGB颜色数组
        gray_color = np.array([0.5, 0.5, 0.5])
        points.point_data['gray_colors'] = np.tile(gray_color, (N, 1))

        # 创建可视化窗口
        plotter = pv.Plotter()
        plotter.add_points(points, scalars='gray_colors', rgb=True, point_size=5)
        # 添加坐标轴并显示网格
        plotter.add_axes(interactive=True)
        plotter.show_grid()
        # 突出显示坐标原点
        origin = pv.Sphere(radius=1.0, center=(0, 0, 0), direction=(0, 0, 1))
        plotter.add_mesh(origin, color='red', show_edges=False)

        # 设置坐标轴位置，将原点固定在左下角
        plotter.view_vector([-1, -1, 0])  # 调整视角
        plotter.reset_camera()  # 重置相机以确保坐标原点在视野中可见
        # 展示结果
        plotter.show(title='PointCloud Gray Visualization')

    elif type == 'rgb':  # 0-255
        # 只显示不为(0, 0, 0)的数据
        rgb_values = A[:, 3:]  # 提取 RGB 值
        non_zero_indices = ~np.all(rgb_values == 0, axis=1)  # 计算非零 RGB 值的索引
        A = A[non_zero_indices]  # 使用非零 RGB 值的索引来过滤点

        # 准备点云数据
        points_rgb = pv.PolyData(A[:, :3])
        points_rgb['rgb'] = A[:, 3:6] / 255  # 归一化到[0, 1]

        # 可视化RGB颜色的点云
        plotter = pv.Plotter()
        plotter.add_points(points_rgb, rgb=True, point_size=5)
        # 添加坐标轴并显示网格
        plotter.add_axes(interactive=True)
        plotter.show_grid()
        # 突出显示坐标原点
        origin = pv.Sphere(radius=1.0, center=(0, 0, 0), direction=(0, 0, 1))
        plotter.add_mesh(origin, color='red', show_edges=False)

        # 设置坐标轴位置，将原点固定在左下角
        plotter.view_vector([-1, -1, 0])  # 调整视角
        plotter.reset_camera()  # 重置相机以确保坐标原点在视野中可见
        # 展示结果
        plotter.show(title='Point Cloud with RGB Colors')

    elif type == 'semantic':
        if use_path:
            B = np.load(input_B)
        else:
            B = input_B
        B = B.reshape(N, -1)

        A = A[B.reshape(-1) > 0]
        B = B[B.reshape(-1) > 0]

        # # 手动指定前三个类别的颜色：红、黄、蓝
        # manual_colors = np.array([
        #     [0, 0, 1],  # 蓝色
        #     [1, 0, 0],  # 红色
        #     [0, 1, 0]  # 绿色
        # ])
        #
        # # 使用调色板为其余类别生成颜色
        # num_classes = 51
        # colors = plt.get_cmap('Set3', num_classes)  # 选用Set3调色板
        # generated_colors = [colors(i / num_classes) for i in range(num_classes)]
        # generated_colors = np.array(generated_colors)[:, :3]  # 仅取RGB，忽略透明度
        #
        # # 合并手动指定的颜色和生成的颜色
        # # 确保手动指定的颜色覆盖前三个类别
        # semantic_colors = np.vstack((manual_colors, generated_colors[3:num_classes]))
        #
        # # 准备语义类别的点云颜色
        # semantic_class_colors = np.array([semantic_colors[i[0]] for i in B])

        # 为每个语义随机分配一个唯一的颜色
        semantic_class_colors = np.random.rand(51, 3)  # 生成随机颜色
        semantic_class_colors = semantic_class_colors[B.flatten()]  # 为每个实例分配颜色

        # 创建点云数据并应用颜色
        points_semantic = pv.PolyData(A[:, :3])
        points_semantic['semantic_colors'] = semantic_class_colors

        # 可视化语义类别的点云
        plotter = pv.Plotter()
        plotter.add_points(points_semantic, rgb=True, point_size=5)
        # 添加坐标轴并显示网格
        plotter.add_axes(interactive=True)
        plotter.show_grid()
        # 突出显示坐标原点
        origin = pv.Sphere(radius=1.0, center=(0, 0, 0), direction=(0, 0, 1))
        plotter.add_mesh(origin, color='red', show_edges=False)

        # 设置坐标轴位置，将原点固定在左下角
        plotter.view_vector([-1, -1, 0])  # 调整视角
        plotter.reset_camera()  # 重置相机以确保坐标原点在视野中可见
        # 展示结果
        plotter.show(title='Point Cloud with Semantic Labels')

    elif type == 'instance':
        if use_path:
            B = np.load(input_B)
        else:
            B = input_B
        B = B.reshape(N, -1)

        A = A[B.reshape(-1) != 0]
        B = B[B.reshape(-1) != 0]

        # 创建点云数据
        points = pv.PolyData(A[:, :3])

        # 为每个实例分配一个唯一的颜色
        instance_colors = np.random.rand(np.max(B)+1, 3)  # 生成随机颜色
        instance_colors = instance_colors[B.flatten()]  # 为每个实例分配颜色

        # 将颜色添加到点云数据中
        points.point_data['instance_colors'] = instance_colors

        # 创建可视化窗口
        plotter = pv.Plotter()
        plotter.add_points(points, scalars='instance_colors', rgb=True, point_size=5)
        # 添加坐标轴并显示网格
        plotter.add_axes(interactive=True)
        plotter.show_grid()
        # 突出显示坐标原点
        origin = pv.Sphere(radius=1.0, center=(0, 0, 0), direction=(0, 0, 1))
        plotter.add_mesh(origin, color='red', show_edges=False)

        # 设置坐标轴位置，将原点固定在左下角
        plotter.view_vector([-1, -1, 0])  # 调整视角
        plotter.reset_camera()  # 重置相机以确保坐标原点在视野中可见
        # 展示结果
        plotter.show(title='Instance Segmentation Result')

    elif type == 'tsdf':
        if use_path:
            B = np.load(input_B)
        else:
            B = input_B
        B = B.reshape(N, -1)

        A = A[B.reshape(-1) > 0]
        B = B[B.reshape(-1) > 0]

        # 归一化TSDF值到[-1, 1]范围内
        # normalized_tsdf = (B - np.min(B)) / (np.max(B) - np.min(B)) * 2 - 1
        normalized_tsdf = B

        # 创建网格
        grid = pv.PolyData(A[:, :3])  # 使用PolyData而不是UnstructuredGrid

        # 将归一化的TSDF值作为标量数据添加到网格中
        grid.point_data['tsdf'] = normalized_tsdf.flatten()

        # 创建可视化窗口
        plotter = pv.Plotter()
        plotter.add_mesh(grid, scalars='tsdf', cmap='plasma')
        # 添加坐标轴并显示网格
        plotter.add_axes(interactive=True)
        plotter.show_grid()
        # 突出显示坐标原点
        origin = pv.Sphere(radius=1.0, center=(0, 0, 0), direction=(0, 0, 1))
        plotter.add_mesh(origin, color='red', show_edges=False)

        # 设置坐标轴位置，将原点固定在左下角
        plotter.view_vector([-1, -1, 0])  # 调整视角
        plotter.reset_camera()  # 重置相机以确保坐标原点在视野中可见
        # 展示结果
        plotter.show(title='TSDF Visualization')

        # Save the grid to a PLY file
        if out_file is not None:
            grid.save(out_file)

    else:
        pass


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

        dense = torch.full([dim[0], dim[1], dim[2]], default_val, device=device).to(values.dtype)
        if locs.shape[0] > 0:
            dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values

    else:
        dense = torch.full([dim[0], dim[1], dim[2]], float(default_val), device=device).to(values.dtype)
        if locs.shape[0] > 0:
            dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values

    return dense


def label2mesh(coords, tsdf, dim_list, label, save_path):
    tsdf = tsdf.view(-1).detach()
    tsdf_volume = sparse_to_dense_torch(coords.long(), tsdf, dim_list, 1, tsdf.device, final_tsdf=True)  # 最后的tsdf需要用-1和1填充
    tsdf_volume = torch.clamp(tsdf_volume, -1, 1)
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
            label_to_color[label] = np.array([192, 192, 192], dtype=np.uint8)  # 类别0为灰色
        elif label == 1:
            label_to_color[label] = np.array([173, 216, 230], dtype=np.uint8)  # 浅蓝色
        elif label == 2:
            label_to_color[label] = np.array([216, 191, 216], dtype=np.uint8)  # 浅紫色 
        elif label == 3:
            label_to_color[label] = np.array([255, 255, 0], dtype=np.uint8)  # 黄色  
        elif label == 4:
            label_to_color[label] = np.array([0, 255, 0], dtype=np.uint8)  # 绿色   
        else:
            label_to_color[label] = (np.array(colormap(i)[:3]) * 255).astype(np.uint8)
    
    
    # 为每个顶点分配颜色
    vertex_colors = np.array([label_to_color[label] for label in vertex_labels])

    # Step 4: 保存到PLY文件，包括顶点、面片、法线和颜色
    mesh.visual.vertex_colors = vertex_colors
    mesh.export(save_path)

    print("mesh save to ", save_path)

colors_list = [
    [230, 230, 230],  # 浅灰色 (用于标签 0)
    [255, 0, 0],      # 红色
    [0, 255, 0],      # 绿色
    [0, 0, 255],      # 蓝色
    [255, 255, 0],    # 黄色
    [0, 255, 255],    # 青色
    [255, 0, 255],    # 品红
    [255, 165, 0],    # 橙色
    [128, 0, 128],    # 紫色
    [0, 128, 128],    # 蓝绿色
    [128, 128, 0],    # 黄绿色
    [0, 0, 128],      # 深蓝色
    [255, 192, 203],  # 粉红色
    [255, 69, 0],     # 橙红色
    [75, 0, 130],     # 靛蓝色
    [255, 99, 71],    # 番茄色
    [34, 139, 34],    # 森林绿
    [0, 0, 139],      # 暗蓝色
    [184, 134, 11],   # 金黄色
    [255, 215, 0],    # 金色
]

def save_coords_with_labels_to_ply(coords, labels, filename):
    """
    将点云数据和标签保存为 PLY 文件，并为每个标签分配不同的颜色。
    
    Args:
        coords (np.ndarray): 点云坐标，形状为 (N, 3)，每行表示一个点的 (x, y, z)。
        labels (np.ndarray): 标签数组，形状为 (N,)，每个元素对应一个点的标签。
        filename (str): 要保存的 PLY 文件名。
    """
    assert coords.ndim == 2 and coords.shape[1] == 3, "coords 必须是形状为 (N, 3) 的数组"
    assert labels.ndim == 1 and labels.shape[0] == coords.shape[0], "labels 必须是与 coords 匹配的形状"
    
    # 分配颜色
    num_labels = np.max(labels) + 1  # 获取标签数目
    colors = np.zeros((labels.shape[0], 3), dtype=int)  # 初始化颜色数组

    # 颜色分配: 标签 0 使用灰色，其它标签按照 colors_list 分配颜色
    for i in range(labels.shape[0]):
        label = labels[i]
        if label == 0:
            colors[i] = colors_list[0]  # 标签 0 为灰色
        else:
            # 如果标签数大于20，循环使用
            colors[i] = colors_list[(label - 1) % 19 + 1]  # 剩下的标签从索引 1 开始
    
    num_points = coords.shape[0]
    
    # PLY 文件头部
    header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    # 将头部和点数据写入文件
    with open(filename, 'w') as ply_file:
        ply_file.write(header)
        # 将坐标和颜色数据一并写入文件
        for i in range(num_points):
            ply_file.write(f"{coords[i, 0]:.6f} {coords[i, 1]:.6f} {coords[i, 2]:.6f} "
                           f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n")


if __name__ == '__main__':
    path_A = 'datasets/scannet/panoptic_info/scene0000_00_vert.npy'
    path_B = 'datasets/scannet/panoptic_info/scene0000_00_sem_label.npy'
    path_C = 'datasets/scannet/panoptic_info/scene0000_00_ins_label.npy'

    # visualize_mesh(path_A, type='xyz', use_path=True)
    # visualize_mesh(path_A, type='rgb', use_path=True)
    # visualize_mesh(path_A, path_B, type='semantic', use_path=True)
    # visualize_mesh(path_A, path_C, type='instance', use_path=True)
    visualize_mesh(path_A, path_B, type='tsdf', use_path=True)


