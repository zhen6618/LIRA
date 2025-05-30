import numpy as np
from PIL import Image
from collections import Counter
import cv2
from skimage import color
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import pickle
import random
import os
import time
from plyfile import PlyData, PlyElement
from scipy.spatial import ConvexHull, Delaunay

from instruction_template import *
from src.llamafactory.chat.chat_model import run_chat_generate_color, ChatModel
from generate_coco_instruction import *
import numpy as np
import open3d as o3d
import copy
from segment_anything import SamPredictor, sam_model_registry
from sklearn.cluster import KMeans
import gzip

instructions_num = 25  # base: 100, hard: 25

def save_instances_to_ply(instances, filename="combined_instances.ply", selected_inds=None):
    points = []
    colors = []
    
    # 生成不同实例的颜色
    color_palette = [tuple(random.choices(range(256), k=3)) for _ in range(len(instances))]
    
    # 遍历每个实例
    for idx, instance in enumerate(instances):
        if idx not in selected_inds and selected_inds is not None:
            continue
        occ_points = instance['occ']
        
        # 将实例的 occ 点添加到总点列表
        points.append(occ_points)
        
        # 为该实例的所有点分配相同颜色
        instance_color = np.array(color_palette[idx]) / 255.0  # 将颜色标准化到 [0, 1]
        colors.append(np.tile(instance_color, (occ_points.shape[0], 1)))
    
    # 合并所有点和颜色
    points = np.vstack(points)
    colors = np.vstack(colors)
    
    # 创建 Open3D 点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    # 保存为 PLY 文件
    o3d.io.write_point_cloud(filename, point_cloud)
    print(f"点云文件已保存为: {filename}")


def coords_to_bwi(points, image, output_path="marked_image.png", radius=5):   
    
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    
    # 绘制每个点
    for (x, y) in points:
        # 定义圆点的边界框
        left_up_point = (x - radius, y - radius)
        right_down_point = (x + radius, y + radius)
        
        # 绘制红色圆点
        draw.ellipse([left_up_point, right_down_point], fill="red", outline="red")
    
    # 保存带标记点的图像
    image.save(output_path)
    print(f"带标记点的图像已保存为: {output_path}")


def coords_to_bw_image(coords, image_size):
    """
    将二维坐标转换为黑白图像
    :param coords: (N, 2) numpy 数组，表示坐标，类型为 float
    :param image_size: 图像的大小 (height, width)
    :return: 黑白图像
    """
    # 初始化黑色图像
    height, width = image_size
    image = np.zeros((height, width), dtype=np.uint8)

    # 将坐标四舍五入到整数索引
    int_coords = np.round(coords).astype(int)

    # 将坐标范围限制在图像大小范围内
    int_coords[:, 0] = np.clip(int_coords[:, 0], 0, width - 1)
    int_coords[:, 1] = np.clip(int_coords[:, 1], 0, height - 1)

    # 将对应坐标的位置设为白色 (255)
    for x, y in int_coords:
        image[y, x] = 255

    return image

# 由坐标和rgb生成点云ply
def save_point_cloud_to_ply(coords, colors, filename):
    # 创建顶点数组，包含位置和颜色
    vertices = np.array(
        [
            (*coords[i], colors[i][0], colors[i][1], colors[i][2])
            for i in range(coords.shape[0])
        ],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    )

    # 创建 PlyElement
    ply_element = PlyElement.describe(vertices, 'vertex')

    # 写入 PLY 文件
    PlyData([ply_element], text=True).write(filename)

def load_ply(file_path):
    """
    读取 PLY 文件并返回点云坐标和颜色
    :param file_path: PLY 文件路径
    :return: 点云坐标和颜色 (points, colors)
    """
    ply_data = PlyData.read(file_path)
    vertex_data = ply_data['vertex']
    points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    colors = np.vstack([vertex_data['red'], vertex_data['green'], vertex_data['blue']]).T
    return points, colors

def point_cloud_to_image(points, colors, intrinsic_matrix, extrinsic_matrix, image_size):
    """
    将点云投影到图像上
    :param points: (N, 3) 点云坐标
    :param colors: (N, 3) 点云颜色 (RGB)
    :param intrinsic_matrix: (3, 3) 相机内参矩阵
    :param extrinsic_matrix: (4, 4) 相机外参矩阵（包含旋转和平移）
    :param image_size: 图像的大小 (height, width)
    :return: 生成的图像
    """
    # 点云转换到相机坐标系
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))  # 变为齐次坐标
    points_camera = extrinsic_matrix @ points_homogeneous.T  # (4, N)
    points_camera = points_camera[:3, :]  # 去掉齐次坐标 (3, N)

    # 使用内参将三维点投影到二维图像平面
    uvw = intrinsic_matrix @ points_camera  # (3, N)
    uvs = uvw[:2, :] / uvw[2, :]  # 归一化，得到图像坐标 (2, N)

    # 初始化图像
    height, width = image_size
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # 将点投影到图像上
    count = 0
    for i in range(uvs.shape[1]):
        u, v = int(uvs[0, i]), int(uvs[1, i])
        if 0 <= u < width and 0 <= v < height:  # 判断点是否在图像范围内
            count += 1
            image[v, u] = colors[i]  # 使用点云的颜色信息绘制点

    return image


def compute_3d_bounding_box(points):
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
 
def generate_instance_data():
    root_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data/zhouzhen05/Data/'

    mode = 'val'
    chat_model = ChatModel()
    print(f'generate_color_label/scannetv2_{mode}.txt')

    with open(f'generate_color_label/scannetv2_{mode}.txt', 'r') as split_file:
        for scene_name_i, scene_name in enumerate(split_file):
            print("tqdm_scene: ", scene_name_i, "/ 1201", scene_name)
            scene_name = scene_name.strip()
            start_time = time.time()

            if os.path.exists(f'/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/grounding_scene_instance_infos/{scene_name}_instance.pkl'):
                print(scene_name, 'exists')
                continue

            # ******************************************************************* 读数据 *******************************************************************
            rgb_path = root_path + 'all_tsdf_9_1/' + scene_name + '/full_rgb_layer0.npz'  # voxel_size: 0.04 
            rgb = np.load(rgb_path, allow_pickle=True)
            rgb = rgb.f.arr_0
            # print(rgb.shape)  # (270, 265, 120, 3)

            sem_path = root_path + 'all_tsdf_9_1/' + scene_name + '/full_semantic_layer_interpolate0.npz'  # voxel_size: 0.04 
            sem = np.load(sem_path, allow_pickle=True)
            sem = sem.f.arr_0
            # print(sem.shape)  # (270, 265, 120)

            ins_path = root_path + 'all_tsdf_9_1/' + scene_name + '/full_instance_layer_interpolate0.npz'  # voxel_size: 0.04 
            ins = np.load(ins_path, allow_pickle=True)
            ins = ins.f.arr_0
            # print(ins.shape)  # (270, 265, 120)

            # ******************************************************************* 提取实例信息 *******************************************************************
            # 通过rgb去除黑色点来去除无效点
            black_index = rgb != np.array([0, 0, 0])
            coords = np.argwhere(black_index[..., 0])
            rgb = rgb[black_index[..., 0]]  # (N, 3)
            sem = sem[black_index[..., 0]]  # (N,)
            ins = ins[black_index[..., 0]]  # (N,)

            unique_instances = np.unique(ins)
            # # 定义常用的颜色
            # common_colors = [
            #     [192, 64, 0],    # Mahogany 红木色
            #     [128, 0, 32],    # Burgundy 酒红色
            #     [0, 255, 0],     # Green 绿色
            #     [255, 253, 208], # Cream 奶油色
            #     [184, 115, 51],  # Copper 铜色
            #     [165, 42, 42],   # Brown 棕色
            #     [255, 0, 0],     # Red 红色
            #     [128, 128, 0],   # Olive 橄榄色
            #     [196, 64, 64],   # Reddish 浅红色
            #     [255, 229, 180], # Peach 桃色
            #     [255, 255, 255], # White 白色
            #     [255, 215, 0],   # Gold 金色
            #     [0, 0, 255],     # Blue 蓝色
            #     [210, 180, 140], # Tan 棕褐色
            #     [255, 165, 0],   # Orange 橙色
            #     [0, 128, 128],   # Teal 蓝绿色
            #     [128, 0, 128],   # Purple 紫色
            #     [128, 128, 128], # Gray 灰色
            #     [245, 245, 220], # Beige 米色
            #     [255, 192, 203], # Pink 粉色
            #     [255, 255, 0],   # Yellow 黄色
            #     [192, 192, 192], # Silver 银色
            #     [128, 0, 0],     # Maroon 栗色
            #     [0, 0, 0],       # Black 黑色
            # ]

            # color_dict = {0: 'Mahogany', 1: 'Burgundy', 2: 'Green', 3: 'Cream', 4: 'Copper', 5: 'Brown',
            #             6: 'Red', 7: 'Olive', 8: 'Reddish', 9: 'Peach', 10: 'White', 11: 'Gold', 12: 'Blue', 
            #             13: 'Tan', 14: 'Orange', 15: 'Teal', 16: 'Purple', 17: 'Gray', 18: 'Beige', 19: 'Pink', 
            #             20: 'Yellow', 21: 'Silver', 22: 'Maroon', 23: 'Black'}

            # 1-wall  2-floor  3-cabinet  4-bed  5-chair
            # 6-sofa  7-table  8-door  9-window  10-bookshelf
            # 11-picture  12-counter  14-desk  16-curtain 24-refrigerator
            # 28-shower curtain  33-toilet  34-sink  36-bathtub  39-otherfurniture

            valid_semantic_labels = [3, 4, 5, 6, 7, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36]
            class_name_dict = {3: 'cabinet', 4: 'bed', 5: 'chair', 
                            6: 'sofa', 7: 'table', 10: 'bookshelf', 
                            11: 'picture', 12: 'counter', 14: 'desk', 16: 'curtain', 24: 'refrigerator',
                            28: 'shower curtain', 33: 'toilet', 34: 'sink', 36: 'bathtub'}
            
            # with open("generate_color_label/mapping.json", 'r', encoding='utf-8') as mapping_file:
            #     label_2d_mapping = json.load(mapping_file)

            # *********************************  基于VLM(Qwen2-VL)获取颜色 *********************************    
            with open(root_path + f'all_tsdf_9_1/fragments_{mode}.pkl', 'rb') as file:
                instance_data_all = pickle.load(file)

            instance_data = [item for item in instance_data_all if item.get("scene") == scene_name]
            all_image_ids = [image_id for item in instance_data for image_id in item["image_ids"]]
            
            instance_results = []
            for instance_id in unique_instances:
                # 获取当前实例的voxel布尔掩码
                instance_mask = (ins == instance_id)
                
                # 找到该实例的所有类别标签
                instance_sem = sem[instance_mask]
                
                # 找到该实例的最常见类别标签
                most_common_label = Counter(instance_sem).most_common(1)[0][0]
                if most_common_label not in valid_semantic_labels:
                    continue
                sem_name = class_name_dict[most_common_label]
                print(instance_id, '/', len(unique_instances))  # 显示进度

                # 计算3D Bounding Box (x, y, z, w, h, l, angle)
                instance_coords = coords[instance_mask]
                bbox_3d = compute_3d_bounding_box(instance_coords)  # 最小包围框 (x, y, z, w, h, l, angle)
                
                # 获取当前实例的所有RGB值
                instance_rgb = rgb[instance_mask]

                # if sem_name == 'curtain':
                #     # save_point_cloud_to_ply(instance_coords, instance_rgb, "curtain.ply")
                    
                # 图片处理
                match_colors = []  # 存储所有图像对应的颜色
                # vlm_generate_color_count = 0
                for img_id in all_image_ids:
                    # print(img_id)
                    img_path = root_path + f'scans/{scene_name}/color/color_{img_id}.jpg'
                    image = Image.open(img_path)
                    image_width, image_height = image.size

                    intrinsics = np.loadtxt(root_path + f'scans/{scene_name}/intrinsic/intrinsic_color.txt', delimiter=' ')[:3, :3]
                    intrinsics = intrinsics.astype(np.float32)

                    pose_path = root_path + f'scans/{scene_name}/pose/pose_{img_id}.txt'
                    extrinsics = np.loadtxt(pose_path)

                    # 直接用ply测试
                    # points, colors = load_ply("scans/scene0000_00/scene0000_00_vh_clean_2.ply")  # 读取点云数据
                    # image_test = point_cloud_to_image(points, colors, intrinsics, extrinsics, (image_height, image_width))
                    # cv2.imwrite("dierct.png", image_test)

                    vol_origin = instance_data[0]['vol_origin']

                    # Convert intrinsics and extrinsics matrices to a single projection matrix
                    proj_mat = np.linalg.inv(extrinsics)
                    proj_mat[:3, :4] = intrinsics @ proj_mat[:3, :4]

                    # 3D点转换成齐次坐标
                    pixel_coordinates_in_view, cur_z = projection(instance_coords, vol_origin, proj_mat, image_height, image_width)

                    if len(pixel_coordinates_in_view) > 50: 
                        # print("vlm_generate_color_count: ", vlm_generate_color_count)
                        # if vlm_generate_color_count >= 10:  # 最多用10张判断
                        #     print('------------------------------   vlm_generate_color_count >= 10   ------------------------------')
                        #     break
                        # vlm_generate_color_count += 1

                        # bw_image = coords_to_bw_image(pixel_coordinates_in_view, (image_height, image_width))
                        # cv2.imwrite('bw_image.png', bw_image)
                        # imgae_copy = copy.deepcopy(image)
                        # coords_to_bwi(pixel_coordinates_in_view, imgae_copy, output_path="marked_bw_image.png")

                        """******************************************************************  直接用3D点来构造2D mask start ******************************************************************"""
                        """ 去除其他实例的遮挡 """
                        # 1. 剔除当前实例
                        all_str = np.ascontiguousarray(coords).view(np.dtype((np.void, coords.dtype.itemsize * coords.shape[1])))
                        cur_str = np.ascontiguousarray(instance_coords).view(np.dtype((np.void, instance_coords.dtype.itemsize * instance_coords.shape[1])))
                        other_str = np.setdiff1d(all_str, cur_str)  # 使用 np.setdiff1d 找出 all_str 中没有出现在 cur_str 中的位置
                        other_ins_coords = other_str.view(coords.dtype).reshape(-1, coords.shape[1])

                        # 2. 剔除遮挡
                        other_pixel_coordinates_in_view, other_z = projection(other_ins_coords, vol_origin, proj_mat, image_height, image_width)  # other instance的2D投影
                        
                        # 计算距离矩阵，并找到距离在固定像素以内的点
                        min_occluded_distance = 20
                        occluded_distances = np.sqrt(((other_pixel_coordinates_in_view[:, np.newaxis, :] - pixel_coordinates_in_view[np.newaxis, :, :]) ** 2).sum(axis=2))

                        close_points_mask = occluded_distances <= min_occluded_distance

                        z_diff_mask = other_z[:, np.newaxis] < cur_z

                        remove_mask = (close_points_mask & z_diff_mask).any(axis=0)

                        pixel_coordinates_in_view = pixel_coordinates_in_view[~remove_mask]

                        # # test
                        # imgae_copy = copy.deepcopy(image)
                        # coords_to_bwi(pixel_coordinates_in_view, imgae_copy, output_path="marked_bw_image_2.png")   
                        
                        if len(pixel_coordinates_in_view) >= 50: 
                            max_2d_mask = np.zeros((image_width, image_height), dtype=bool)

                            pixel_coordinates_in_view = np.clip(np.round(pixel_coordinates_in_view).astype(int), [0, 0], [image_width-1, image_height-1])
                            max_2d_mask[pixel_coordinates_in_view[:, 0], pixel_coordinates_in_view[:, 1]] = True                 
                            
                            """******************************************************************  直接用3D点来构造2D mask end ******************************************************************"""

                            # 为避免直接填充黑色，导致黑色物体无法判断，选择截取包含目标mask的最小矩形框
                            rows, cols = np.where(max_2d_mask == True)
                            top_left = (min(rows), min(cols))  # 最上面的那个True的坐标
                            bottom_right = (max(rows), max(cols))  # 最下面的那个True的坐标
                            matrix_height = bottom_right[0] - top_left[0] + 1
                            matrix_width = bottom_right[1] - top_left[1] + 1
                            new_matrix_height = int(matrix_height * 1.5)  # 扩展
                            new_matrix_width = int(matrix_width * 1.5)
                            # 计算新的中心点位置
                            center_row = top_left[0] + matrix_height // 2
                            center_col = top_left[1] + matrix_width // 2
                            
                            # 计算新的左上角和右下角坐标
                            new_top_left_row = max(center_row - new_matrix_height // 2, 0)
                            new_top_left_col = max(center_col - new_matrix_width // 2, 0)
                            new_bottom_right_row = min(center_row + new_matrix_height // 2, max_2d_mask.shape[0]-1)
                            new_bottom_right_col = min(center_col + new_matrix_width // 2, max_2d_mask.shape[1]-1)

                            max_2d_mask[new_top_left_row:new_bottom_right_row+1,
                                        new_top_left_col:new_bottom_right_col+1] = True

                            max_2d_mask = np.transpose(max_2d_mask, (1, 0))

                            image_array = np.array(image)
                            selected_image_array = np.zeros_like(image_array)  
                            selected_image_array[max_2d_mask] = image_array[max_2d_mask]
                            # selected_image_array = image_array[new_top_left_row:new_bottom_right_row+1,
                            #                                    new_top_left_col:new_bottom_right_col+1, :]

                            # selected_image = Image.fromarray(selected_image_array.astype('uint8'))   
                            # selected_image.save('selected_image.png')  

                            # 使用VLM进行颜色打标签
                            match_color = run_chat_generate_color(chat_model=chat_model, object_name=sem_name, image_array=selected_image_array)
                            match_colors.append(match_color)

                if len(match_colors) == 0:
                    continue
                match_object_color = Counter(match_colors).most_common(1)[0][0]

                instance_results.append({
                    'semantic label': class_name_dict[most_common_label],
                    'color': match_object_color,
                    'bbox_3d': bbox_3d,  # 最小包围框 (x, y, z, w, h, l, angle) (7, )
                    'occ': instance_coords,  # voxel坐标 (N, 3)
                })

            with open(f'/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/grounding_scene_instance_infos/{scene_name}_instance.pkl', 'wb') as file:
                pickle.dump(instance_results, file)

            print("file save to ", f'/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/grounding_scene_instance_infos/{scene_name}_instance.pkl')
            elapsed_time = (time.time() - start_time) / 60
            print(f"{scene_name} time: {elapsed_time:.6f} min.")

        print('Done')

def generate_grounding_3D_data():

    root_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/'

    with open('generate_color_label/scannetv2_val.txt', 'r') as split_file:  
        for scene_name_i, scene_name in enumerate(split_file):
            # print("tqdm_scene: ", scene_name_i, "/ 1201")
            scene_name = scene_name.strip()
            # print("Processing scene_name: ", scene_name)

            # if not scene_name in ['scene0000_00']:
            #     continue

            with open(root_path + f'grounding_scene_instance_infos/{scene_name}_instance.pkl', 'rb') as file:
                instance_data = pickle.load(file)
            if len(instance_data) == 0:
                continue
            
            # save instance_data to .ply
            # save_instances_to_ply(instance_data, filename="combined_instances.ply")

            # *******************  预信息  *******************
            classes = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'bookshelf', 
                       'picture', 'counter', 'desk', 'curtain', 'refrigerator', 
                       'shower curtain', 'toilet', 'sink', 'bathtub']
            classes_plural = ['cabinets', 'beds', 'chairs', 'sofas', 'tables', 'bookshelves', 
                              'pictures', 'counters', 'desks', 'curtains', 'refrigerators', 
                              'shower curtains', 'toilets', 'sinks', 'bathtubs']

            color_valid_classes = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'desk', 'refrigerator', 'toilet', 'sink', 'bathtub']

            # 当前场景下所有实例的类别集合
            class_all = []
            for _, ins in enumerate(instance_data):
                class_all.append(ins['semantic label'])
            class_all = list(set(class_all))  # 在当前
            class_all_no = [item for item in classes if item not in class_all]

            # 当前场景下所有实例的颜色集合
            color_all = []
            for _, ins in enumerate(instance_data):
                color_all.append(ins['color'])
            color_all = list(set(color_all))

            # *******************  每个场景随机从8个instruction中，随机选择1个问题  *******************
            instruction_choice_templates = ["class", 
                                           "class_color", 
                                           "class_color_distance",
                                           "class_color_size",
                                           "implicit_class_one",
                                           "implicit_class_color",
                                           "implicit_class_size",
                                           "implicit_class_multiple",
                                           ]

            for instructions_i in range(instructions_num):
                random.shuffle(instruction_choice_templates)

                scene_qa = []
                for selected_instruction_template in instruction_choice_templates:
                    print("scene_name: ", scene_name, "selected_instruction_template: ", selected_instruction_template, "instructions_num: ", instructions_num)

                    # 1. *******************  for "class"  *******************
                    if selected_instruction_template == "class":
                        
                        q_choice_instruction = random.choice(class_one)  # 随机选择一个问题模板
                        q_choice_ins = random.choice(class_all)  # 从ins中选择一个类
                        q_choice_ins_plural = classes_plural[classes.index(q_choice_ins)]  # classes_plural
                        q_final = q_choice_instruction.replace(class_one_substitute_word[0], q_choice_ins_plural)  # 将模板中的占位符替换为实际的实例名称
                        
                        # store the whole scene qa
                        ins_i_collection = []
                        for k, ins in enumerate(instance_data):
                            if ins['semantic label'] == q_choice_ins:
                                ins_i_collection.append(k)

                        # candidate
                        candidate_ins_indexs = []
                        for c in range(len(instance_data)):
                            if instance_data[c]['semantic label'] == q_choice_ins:
                                candidate_ins_indexs.append(c)

                        scene_qa.append({
                            'instruction': q_final,
                            'instance ids': ins_i_collection,
                            'candidate': candidate_ins_indexs,
                        })

                        print("class done.")


                    # 2. *******************  for "class_one_color_one"  ******************* 
                    elif selected_instruction_template == "class_color":

                        q_choice_instruction = random.choice(class_one_color_one)  # 随机选择一个问题模板

                        shuffled_instance_data = instance_data.copy()  # 打乱实例列表
                        random.shuffle(shuffled_instance_data)

                        for q_choice_ins in shuffled_instance_data:
                            q_choice_ins_class = q_choice_ins['semantic label']
                            q_choice_ins_plural = classes_plural[classes.index(q_choice_ins_class)]  # classes_plural
                            q_choice_ins_color = q_choice_ins['color']

                            if q_choice_ins_class not in color_valid_classes:
                                continue
                            else:
                                q_final = q_choice_instruction.replace(class_one_color_one_substitute_word[0], q_choice_ins_plural).replace(class_one_color_one_substitute_word[1], q_choice_ins_plural)
                                q_final = q_final.replace(class_one_color_one_substitute_word[2], q_choice_ins_color.lower())

                                # candidate
                                candidate_ins_indexs = []
                                for c in range(len(instance_data)):
                                    if instance_data[c]['semantic label'] == q_choice_ins_class and instance_data[c]['color'] == q_choice_ins_color:
                                        candidate_ins_indexs.append(c)

                                # store the whole scene qa
                                ins_i_collection = []
                                for k, ins in enumerate(instance_data):
                                    if ins['semantic label'] in q_choice_ins_class and ins['color'] in q_choice_ins_color:
                                        ins_i_collection.append(k)

                                scene_qa.append({
                                    'instruction': q_final,
                                    'instance ids': ins_i_collection,
                                    'candidate': candidate_ins_indexs,
                                })

                                print("class_color done.")

                                break


                    # 3. *******************  for "class_color_bbox_distance"  ******************* 
                    elif selected_instruction_template == "class_color_distance":
                        q_choice_instruction = random.choice(class_color_bbox_distance)  # 随机选择一个问题模板
                        # 找到唯一的具有某种color和class的实例
                        shuffled_instance_data = instance_data.copy()  # 打乱实例列表
                        random.shuffle(shuffled_instance_data)
                        
                        unique_ins_flag = False
                        for ins in shuffled_instance_data:
                            current_ins_class = ins['semantic label']  # 获取当前实例的类别和标签
                            current_ins_color = ins['color'] 
                            
                            # 检查是否有其他实例也有相同的类别和标签
                            unique_ins = [other_ins['semantic label'] == current_ins_class and other_ins['color'] == current_ins_color for other_ins in instance_data]
                            
                            if sum(unique_ins) == 1 and current_ins_class in color_valid_classes: # 如果唯一，返回该实例; 检查合法类别
                                unique_ins_flag = True
                                break

                        if unique_ins_flag == True:
                            q_choice_ins_plural = classes_plural[classes.index(current_ins_class)]  # classes_plural
                            q_final = q_choice_instruction.replace(class_color_distance_substitute_word[0], q_choice_ins_plural)
                            q_final = q_final.replace(class_color_distance_substitute_word[2], current_ins_color.lower())

                            other_classes = [other_class for other_class in class_all if other_class != current_ins_class]  # 选择其他类

                            if len(other_classes) > 0:
                                q_choice_class_B = random.choice(other_classes)  # 从其他类中随机选择一个类
                                q_choice_class_B_plural = classes_plural[classes.index(q_choice_class_B)]  # classes_plural
                                q_final = q_final.replace(class_color_distance_substitute_word[1], q_choice_class_B_plural)

                                q_choice_class_B_num = [ins for ins in instance_data if ins['semantic label'] == q_choice_class_B]  # 统计q_choice_class_B实例个数
                                q_choice_class_B_num = len(q_choice_class_B_num)
                                # 选一个距离
                                if q_choice_class_B_num > 1:
                                    random_integer = random.randint(0, 3)
                                    if random_integer == 0:
                                        q_choice_distance = random.choice(class_color_bbox_diatance_howfar_1)
                                        distance_type = "class_color_bbox_diatance_howfar_1"  # 最近
                                    elif random_integer == 1:
                                        q_choice_distance = random.choice(class_color_bbox_diatance_howfar_2)
                                        distance_type = "class_color_bbox_diatance_howfar_2"  # 最远
                                    elif random_integer == 2:
                                        q_choice_distance = random.choice(class_color_bbox_diatance_howfar_3)
                                        distance_type = "class_color_bbox_diatance_howfar_3"  # 第二近
                                    elif random_integer == 3:
                                        q_choice_distance = random.choice(class_color_bbox_diatance_howfar_4)
                                        distance_type = "class_color_bbox_diatance_howfar_4"  # 第二远
                                    q_final = q_final.replace(class_color_distance_substitute_word[3], q_choice_distance)
                                elif q_choice_class_B_num == 1:
                                    random_integer = random.randint(0, 1)
                                    if random_integer == 0:
                                        q_choice_distance = random.choice(class_color_bbox_diatance_howfar_1)
                                        distance_type = "class_color_bbox_diatance_howfar_1"
                                    elif random_integer == 1:
                                        q_choice_distance = random.choice(class_color_bbox_diatance_howfar_2)
                                        distance_type = "class_color_bbox_diatance_howfar_2"
                                    q_final = q_final.replace(class_color_distance_substitute_word[3], q_choice_distance)

                                # store the whole scene qa
                                # 找到匹配的实例索引
                                q_choice_class_A_ids = [index for index, ins in enumerate(instance_data) if ins['semantic label'] == current_ins_class and ins['color'] == current_ins_color]  
                                q_choice_class_B_ids = [index for index, ins in enumerate(instance_data) if ins['semantic label'] == q_choice_class_B]

                                q_distances = [np.linalg.norm(instance_data[q_choice_class_A_ids[0]]['bbox_3d'][0:3] - instance_data[B_id]['bbox_3d'][0:3]) for B_id in q_choice_class_B_ids]  
                                q_distances_sorted_indices = np.argsort(q_distances)

                                ins_i_collection = []
                                if distance_type == "class_color_bbox_diatance_howfar_1":  # 最近
                                    ins_i_collection.append(q_choice_class_B_ids[q_distances_sorted_indices[0]])
                                elif distance_type == "class_color_bbox_diatance_howfar_2":  # 最远
                                    ins_i_collection.append(q_choice_class_B_ids[q_distances_sorted_indices[-1]])
                                elif distance_type == "class_color_bbox_diatance_howfar_3":  # 第二近
                                    ins_i_collection.append(q_choice_class_B_ids[q_distances_sorted_indices[1]])
                                elif distance_type == "class_color_bbox_diatance_howfar_4":  # 第二远
                                    ins_i_collection.append(q_choice_class_B_ids[q_distances_sorted_indices[-2]])

                                # candidate
                                candidate_ins_indexs = []
                                for c in range(len(instance_data)):
                                    if instance_data[c]['semantic label'] == current_ins_class and instance_data[c]['color'] == current_ins_color or \
                                        instance_data[c]['semantic label'] == q_choice_class_B:
                                        candidate_ins_indexs.append(c)

                                scene_qa.append({
                                    'instruction': q_final,
                                    'instance ids': ins_i_collection,
                                    'candidate': candidate_ins_indexs,
                                })

                                print("class_color_distance done.")

                    # 4. *******************  for "class_color_bbox_size"  ******************* 
                    elif selected_instruction_template == "class_color_size":
                        q_choice_instruction = random.choice(class_color_bbox_size)  # 随机选择一个问题模板

                        shuffled_instance_data = instance_data.copy()  # 打乱实例列表
                        random.shuffle(shuffled_instance_data)

                        for q_choice_ins in shuffled_instance_data:
                            q_choice_ins_class = q_choice_ins['semantic label']
                            q_choice_ins_color = q_choice_ins['color']

                            if q_choice_ins_class not in color_valid_classes:
                                continue
                            else:
                                q_choice_ins_class_plural = classes_plural[classes.index(q_choice_ins_class)]  # classes_plural
                                q_final = q_choice_instruction.replace(class_color_bbox_size_substitute_word[0], q_choice_ins_class_plural).replace(class_color_bbox_size_substitute_word[1], q_choice_ins_color.lower())

                                q_choice_ins_num = [ins for ins in instance_data if ins['color'] == q_choice_ins_color and ins['semantic label'] == q_choice_ins_class]  # 统计实例个数
                                q_choice_ins_num = len(q_choice_ins_num)

                                if q_choice_ins_num > 1:
                                    random_integer = random.randint(0, 3)
                                    if random_integer == 0:
                                        q_choice_distance = random.choice(class_color_bbox_size_howsize_1)
                                        distance_type = "class_color_bbox_size_howsize_1"  # 最大
                                    elif random_integer == 1:
                                        q_choice_distance = random.choice(class_color_bbox_size_howsize_2)
                                        distance_type = "class_color_bbox_size_howsize_2"  # 最小
                                    elif random_integer == 2:
                                        q_choice_distance = random.choice(class_color_bbox_size_howsize_3)
                                        distance_type = "class_color_bbox_size_howsize_3"  # 第二大
                                    elif random_integer == 3:
                                        q_choice_distance = random.choice(class_color_bbox_size_howsize_4)
                                        distance_type = "class_color_bbox_size_howsize_4"  # 第二小
                                    q_final = q_final.replace(class_color_bbox_size_substitute_word[2], q_choice_distance)

                                elif q_choice_ins_num == 1:
                                    random_integer = random.randint(0, 1)
                                    if random_integer == 0:
                                        q_choice_distance = random.choice(class_color_bbox_size_howsize_1)
                                        distance_type = "class_color_bbox_size_howsize_1"  # 最大
                                    elif random_integer == 1:
                                        q_choice_distance = random.choice(class_color_bbox_size_howsize_2)
                                        distance_type = "class_color_bbox_size_howsize_2"  # 最小
                                    q_final = q_final.replace(class_color_bbox_size_substitute_word[2], q_choice_distance)

                                # store the whole scene qa
                                q_choice_color_ids = [index for index, ins in enumerate(instance_data) if ins['color'] == q_choice_ins_color and ins['semantic label'] == q_choice_ins_class]  # 找到匹配的实例索引
                                q_sizes = [np.prod(instance_data[size_id]['bbox_3d'][3:6]) for size_id in q_choice_color_ids]
                                q_sizes_sorted_indices = np.argsort(q_sizes)

                                ins_i_collection = []
                                if distance_type == "class_color_bbox_size_howsize_1":  # 最大
                                    ins_i_collection.append(q_choice_color_ids[q_sizes_sorted_indices[-1]])
                                elif distance_type == "class_color_bbox_size_howsize_2":  # 最小
                                    ins_i_collection.append(q_choice_color_ids[q_sizes_sorted_indices[0]])
                                elif distance_type == "class_color_bbox_size_howsize_3":  # 第二大
                                    ins_i_collection.append(q_choice_color_ids[q_sizes_sorted_indices[-2]])
                                elif distance_type == "class_color_bbox_size_howsize_4":  # 第二小
                                    ins_i_collection.append(q_choice_color_ids[q_sizes_sorted_indices[1]])

                                # candidate
                                candidate_ins_indexs = []
                                for c in range(len(instance_data)):
                                    if instance_data[c]['semantic label'] == q_choice_ins_class and instance_data[c]['color'] == q_choice_ins_color:
                                        candidate_ins_indexs.append(c)

                                scene_qa.append({
                                    'instruction': q_final,
                                    'instance ids': ins_i_collection,
                                    'candidate': candidate_ins_indexs,
                                })

                                print("class_color_size done.")

                                break


                    # 5. *******************  for "implicit one class"  ******************* 
                    elif selected_instruction_template == "implicit_class_one":

                        q_choice_ins = random.choice(class_all)  # 从ins中选择一个类
                        q_choice_instruction = random.choice(implicit_class_one[q_choice_ins])  # 随机选择一个问题模板
                        q_final = q_choice_instruction
                        
                        # store the whole scene qa
                        ins_i_collection = []
                        for k, ins in enumerate(instance_data):
                            if ins['semantic label'] == q_choice_ins:
                                ins_i_collection.append(k)

                        # candidate
                        candidate_ins_indexs = []
                        for c in range(len(instance_data)):
                            if instance_data[c]['semantic label'] == q_choice_ins:
                                candidate_ins_indexs.append(c)

                        scene_qa.append({
                            'instruction': q_final,
                            'instance ids': ins_i_collection,
                            'candidate': candidate_ins_indexs,
                        })

                        print("implicit_class_one done.")


                    # 6. *******************  for "implicit one class in color one"  *******************  
                    elif selected_instruction_template == "implicit_class_color":

                        shuffled_instance_data = instance_data.copy()  # 打乱实例列表
                        random.shuffle(shuffled_instance_data)

                        for q_choice_ins in shuffled_instance_data:
                            q_choice_ins_class = q_choice_ins['semantic label']
                            q_choice_ins_color = q_choice_ins['color']

                            if q_choice_ins_class not in color_valid_classes:
                                continue
                            else:
                                q_choice_instruction = random.choice(implicit_class_one[q_choice_ins_class])  # 随机选择一个问题模板
                                q_final = q_choice_instruction.replace(".", "")
                                q_final = q_final + ", and all objects that meet the requirements are " + q_choice_ins_color.lower() + "."

                                # store the whole scene qa
                                ins_i_collection = []
                                for k, ins in enumerate(instance_data):
                                    if ins['semantic label'] == q_choice_ins_class and ins['color'] == q_choice_ins_color:
                                        ins_i_collection.append(k)

                                # candidate
                                candidate_ins_indexs = []
                                for c in range(len(instance_data)):
                                    if instance_data[c]['semantic label'] == q_choice_ins_class and instance_data[c]['color'] == q_choice_ins_color:
                                        candidate_ins_indexs.append(c)

                                scene_qa.append({
                                    'instruction': q_final,
                                    'instance ids': ins_i_collection,
                                    'candidate': candidate_ins_indexs,
                                })

                                print("implicit_class_color done.")

                                break


                    # 7. *******************  for "implicit one class in bbox"  *******************
                    elif selected_instruction_template == "implicit_class_size":

                        q_choice_ins = random.choice(class_all)  # 从ins中选择一个类
                        q_choice_instruction = random.choice(implicit_class_one[q_choice_ins])  # 随机选择一个问题模板

                        q_choice_ins_class_num = [ins for ins in instance_data if ins['semantic label'] == q_choice_ins]  # 统计实例个数
                        q_choice_ins_class_num = len(q_choice_ins_class_num)

                        if q_choice_ins_class_num > 1:
                            random_integer = random.randint(0, 3)
                            if random_integer == 0:
                                q_choice_distance = random.choice(class_color_bbox_size_howsize_1)
                                distance_type = "class_color_bbox_size_howsize_1"  # 最大
                            elif random_integer == 1:
                                q_choice_distance = random.choice(class_color_bbox_size_howsize_2)
                                distance_type = "class_color_bbox_size_howsize_2"  # 最小
                            elif random_integer == 2:
                                q_choice_distance = random.choice(class_color_bbox_size_howsize_3)
                                distance_type = "class_color_bbox_size_howsize_3"  # 第二大
                            elif random_integer == 3:
                                q_choice_distance = random.choice(class_color_bbox_size_howsize_4)
                                distance_type = "class_color_bbox_size_howsize_4"  # 第二小

                            q_final = q_choice_instruction.replace(".", "")
                            q_final = q_final + ", and find the " + q_choice_distance + " object among those that meet the requirements."

                        elif q_choice_ins_class_num == 1:
                            random_integer = random.randint(0, 1)
                            if random_integer == 0:
                                q_choice_distance = random.choice(class_color_bbox_size_howsize_1)
                                distance_type = "class_color_bbox_size_howsize_1"  # 最大
                            elif random_integer == 1:
                                q_choice_distance = random.choice(class_color_bbox_size_howsize_2)
                                distance_type = "class_color_bbox_size_howsize_2"  # 最小
                            
                            q_final = q_choice_instruction.replace(".", "")
                            q_final = q_final + ", and find the " + q_choice_distance + " object among those that meet the requirements."

                        # store the whole scene qa
                        q_choice_class_ids = [index for index, ins in enumerate(instance_data) if ins['semantic label'] == q_choice_ins]  # 找到匹配的实例索引
                        q_sizes = [np.prod(instance_data[size_id]['bbox_3d'][3:6]) for size_id in q_choice_class_ids]
                        q_sizes_sorted_indices = np.argsort(q_sizes)

                        ins_i_collection = []
                        if distance_type == "class_color_bbox_size_howsize_1":  # 最大
                            ins_i_collection.append(q_choice_class_ids[q_sizes_sorted_indices[-1]])
                        elif distance_type == "class_color_bbox_size_howsize_2":  # 最小
                            ins_i_collection.append(q_choice_class_ids[q_sizes_sorted_indices[0]])
                        elif distance_type == "class_color_bbox_size_howsize_3":  # 第二大
                            ins_i_collection.append(q_choice_class_ids[q_sizes_sorted_indices[-2]])
                        elif distance_type == "class_color_bbox_size_howsize_4":  # 第二小
                            ins_i_collection.append(q_choice_class_ids[q_sizes_sorted_indices[1]])

                        # candidate
                        candidate_ins_indexs = []
                        for c in range(len(instance_data)):
                            if instance_data[c]['semantic label'] == q_choice_ins:
                                candidate_ins_indexs.append(c)

                        scene_qa.append({
                            'instruction': q_final,
                            'instance ids': ins_i_collection,
                            'candidate': candidate_ins_indexs,
                        })

                        print("implicit_class_size done.")


                    # 8. *******************  for "implicit more classes"  *******************
                    elif selected_instruction_template == "implicit_class_multiple":

                        q_choice_instructions = random.choice(implicit_more_classes) 
                        q_final = q_choice_instructions["Question"]

                        # store the whole scene qa
                        ins_i_collection = []
                        for k, ins in enumerate(instance_data):
                            if ins['semantic label'] in q_choice_instructions["Answer"]:
                                ins_i_collection.append(k)

                        # candidate
                        candidate_ins_indexs = []
                        for c in range(len(instance_data)):
                            if instance_data[c]['semantic label'] in q_choice_instructions["Answer"]:
                                candidate_ins_indexs.append(c)

                        scene_qa.append({
                            'instruction': q_final,
                            'instance ids': ins_i_collection,
                            'candidate': candidate_ins_indexs,
                        })

                        print("implicit_class_multiple done.")

                    
                    if len(scene_qa) == 0:
                        continue
                    else:
                        with open(root_path + f'grounding_scene_qa_infos_hard/{scene_name}_{instructions_i}_qa.pkl', 'wb') as scene_qa_file:
                            pickle.dump(scene_qa, scene_qa_file)
                        break

            print("Finish", scene_name)
            print("-------------------------------")


def save_img_test(data):
    color_map = {
        0: (0, 0, 0),       # 黑色
        1: (255, 0, 0),     # 红色
        3: (0, 255, 0),     # 绿色
        4: (0, 0, 255),     # 蓝色
        7: (255, 255, 0),   # 黄色
        9: (255, 0, 255),   # 紫色
        21: (0, 255, 255)   # 青色
    }

    # 创建彩色图像
    color_image = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)

    # 将 unique 值映射到颜色
    for value, color in color_map.items():
        color_image[data == value] = color

    # 保存图像
    cv2.imwrite("colored_image.png", color_image)

def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    import csv
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    # if ints convert 
    mapping = {int(k): v for k, v in mapping.items()}
    return mapping

def map_label_image(image, label_mapping):
    mapped = np.copy(image)
    for k, v in label_mapping.items():
        mapped[image == k] = v
    return mapped.astype(np.uint8)


def projection(points, vol_origin, proj_mat, image_height, image_width):
    # 3D点转换成齐次坐标
    voxel_size = 0.04
    points = points * voxel_size + vol_origin
    points_3d_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))  # (n, 4)

    points_2d_homogeneous = proj_mat @ points_3d_homogeneous.T
    points_2d = points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]  # 归一化齐次坐标，得到图像中的像素坐标
    pixel_coordinates = points_2d.T  # 转置以得到形状为 (n, 2) 的像素坐标

    # 过滤掉投影到相机视图外的点
    in_view_mask = (points_2d_homogeneous[2, :] > 0) & \
                (pixel_coordinates[:, 0] >= 0) & (pixel_coordinates[:, 0] < image_width) & \
                (pixel_coordinates[:, 1] >= 0) & (pixel_coordinates[:, 1] < image_height)
    pixel_coordinates_in_view = pixel_coordinates[in_view_mask]  # (n, 2)

    z = points_2d_homogeneous[2, :][in_view_mask]

    return pixel_coordinates_in_view, z


def generate_VLM_grounding_data():
    root_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/'
    root_path_orinal_data = '/root/paddlejob/workspace/env_run/zhouzhen05/Data/zhouzhen05/Data/'
    # root_path_2d_annotations = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_2d_annotations/'

    mode = 'val'
    # scene_file = f'generate_color_label/scannetv2_{mode}.txt'
    scene_file = 'generate_color_label/selected_scenes_3.txt'

    sam = sam_model_registry["vit_h"](checkpoint="generate_color_label/sam_vit_h_4b8939.pth")
    sam.to(device="cuda:1")
    sam_predictor = SamPredictor(sam)

    print(scene_file, mode, sam.device)
    time.sleep(5)
    t_start = time.time()

    with open(scene_file, 'r') as split_file:
        for scene_name_i, scene_name in enumerate(split_file):
            print("**************************************************************** tqdm_scene: ", scene_name_i, "****************************************************************")
            scene_name = scene_name.strip()
            print("Processing scene_name: ", scene_name, "time: ", time.time() - t_start, "s.")

            # if not scene_name in ['scene0296_00']:
            #         continue

            for instructions_i in range(instructions_num):
                print("instructions_i: ", instructions_i)
                # if os.path.exists(root_path + f'grounding_scene_rs_infos/{scene_name}_{instructions_i}_reasoning_seg_data.pkl'):
                #     print(f'{scene_name}_{instructions_i}', 'exists')
                #     continue

                """ 获取所有实例的数据 start """
                rgb_path = root_path_orinal_data + 'all_tsdf_9_1/' + scene_name + '/full_rgb_layer0.npz'  # voxel_size: 0.04 
                rgb = np.load(rgb_path, allow_pickle=True)
                rgb = rgb.f.arr_0

                # 通过rgb去除黑色点来去除无效点
                black_index = rgb != np.array([0, 0, 0])
                all_ins_coords = np.argwhere(black_index[..., 0])
                """ 获取所有实例的数据 end """    
                with open(root_path + f'grounding_scene_instance_infos/{scene_name}_instance.pkl', 'rb') as file:
                    instance_data = pickle.load(file)
                if len(instance_data) == 0:
                    continue

                with open(root_path + f'grounding_scene_qa_infos/{scene_name}_{instructions_i}_qa.pkl', 'rb') as file:
                    scene_qa = pickle.load(file)

                # save_instances_to_ply(instance_data, filename="combined_instances.ply", selected_inds=scene_qa[0]['instance ids'])

                with open(f'/root/paddlejob/workspace/env_run/zhouzhen05/Data/zhouzhen05/Data/all_tsdf_9_1/fragments_{mode}.pkl', 'rb') as file:
                    img_data_all = pickle.load(file)
                img_data = [item for item in img_data_all if item.get("scene") == scene_name]
                if len(img_data) == 0:
                    continue
                all_image_ids = [image_id for item in img_data for image_id in item["image_ids"]]

                # with open("generate_color_label/mapping.json", 'r', encoding='utf-8') as mapping_file:
                #     label_2d_mapping = json.load(mapping_file)

                class_name_dict = {3: 'cabinet', 4: 'bed', 5: 'chair', 
                    6: 'sofa', 7: 'table', 10: 'bookshelf', 
                    11: 'picture', 12: 'counter', 14: 'desk', 16: 'curtain', 24: 'refrigerator',
                    28: 'shower curtain', 33: 'toilet', 34: 'sink', 36: 'bathtub'}

                qa = scene_qa[0]  # 每个场景就一个qa
                print("qa: ", qa)
                instruction = qa['instruction']
                instance_ids = qa['instance ids']  # 3D answer
                candidate_ins_ids = np.array(qa['candidate'])  # 3D candidate
                grounding_2d_data = []

                candidate_count_dict = {}
                for candidate_ins_id in candidate_ins_ids:
                    candidate_count_dict[str(candidate_ins_id)] = 0

                # ************************  staring process  *********************
                # 对于每张图像，将实例id映射到2D mask
                for img_id_count, img_id in enumerate(all_image_ids):
                    print(img_id_count, "/", len(all_image_ids))
                    candidate_ins_2d_all = []
                    img_path = root_path_orinal_data + f'scans/{scene_name}/color/color_{img_id}.jpg'
                    image = Image.open(img_path)
                    image_width, image_height = image.size

                    # ins_2d_path = root_path_2d_annotations + f'{scene_name}/instance-filt/{img_id}.png'
                    # ins_2d_img = cv2.imread(ins_2d_path, cv2.IMREAD_GRAYSCALE)  # 8位

                    # cls_2d_path = root_path_2d_annotations + f'{scene_name}/label-filt/{img_id}.png'
                    # cls_2d_img = cv2.imread(cls_2d_path, cv2.IMREAD_UNCHANGED)  # 16位
                    # # import imageio
                    # # import skimage.transform as sktf
                    # # image = np.array(imageio.imread(cls_2d_path))
                    # # image = sktf.resize(image, [image_height, image_width], order=0, preserve_range=True)
                    # # label_map = read_label_mapping("/root/paddlejob/workspace/env_run/zhouzhen05/Code/DRecon/datasets/scannet/meta_data/scannetv2-labels.combined.tsv", label_from='id', label_to='nyu40id')
                    # # mapped_image = map_label_image(image, label_map)
                    # # imageio.imwrite("label_BPNet.png", mapped_image)

                
                    intrinsics = np.loadtxt(root_path_orinal_data + f'scans/{scene_name}/intrinsic/intrinsic_color.txt', delimiter=' ')[:3, :3]
                    intrinsics = intrinsics.astype(np.float32)

                    pose_path = root_path_orinal_data + f'scans/{scene_name}/pose/pose_{img_id}.txt'
                    extrinsics = np.loadtxt(pose_path)

                    vol_origin = img_data[0]['vol_origin']

                    # Convert intrinsics and extrinsics matrices to a single projection matrix
                    proj_mat = np.linalg.inv(extrinsics)
                    proj_mat[:3, :4] = intrinsics @ proj_mat[:3, :4]

                    for candidate_ins_id in candidate_ins_ids:  # 3D candidate
                        pixel_coordinates_in_view, cur_z = projection(instance_data[candidate_ins_id]['occ'], vol_origin, proj_mat, image_height, image_width)

                        # bw_image = coords_to_bw_image(pixel_coordinates_in_view, (image_height, image_width))
                        # cv2.imwrite('bw_image.png', bw_image)
                        # imgae_copy = copy.deepcopy(image)
                        # coords_to_bwi(pixel_coordinates_in_view, imgae_copy, output_path="marked_bw_image.png")

                        # # 保存查看
                        # # mapped_image_unique = np.unique(mapped_image) 
                        # # cls_2d_img_unique = np.unique(cls_2d_img)
                        # ins_2d_img_unique = np.unique(ins_2d_img)
                        # alpha = 0.5
                        # original_image = Image.open(img_path).convert("RGBA")
                        # color_map = {uid: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), int(255 * alpha)) for uid in ins_2d_img_unique}
                        # mask_image = Image.new("RGBA", (image_width, image_height))
                        # for uid, color in color_map.items():
                        #     instance_mask = (ins_2d_img == uid)
                        #     mask_layer = Image.new("RGBA", (image_width, image_height), color)
                        #     mask_image.paste(mask_layer, (0, 0), Image.fromarray((instance_mask * 255).astype(np.uint8)))
                        # overlayed_image = Image.alpha_composite(original_image, mask_image)
                        # overlayed_image.convert("RGB").save("instance_mask.png")

                        if len(pixel_coordinates_in_view) > 50: 
                            """ 去除其他实例的遮挡 """
                            # 1. 剔除当前实例
                            current_ins_coords = instance_data[candidate_ins_id]['occ']
                            all_str = np.ascontiguousarray(all_ins_coords).view(np.dtype((np.void, all_ins_coords.dtype.itemsize * all_ins_coords.shape[1])))
                            cur_str = np.ascontiguousarray(current_ins_coords).view(np.dtype((np.void, current_ins_coords.dtype.itemsize * current_ins_coords.shape[1])))
                            other_str = np.setdiff1d(all_str, cur_str)  # 使用 np.setdiff1d 找出 all_str 中没有出现在 cur_str 中的位置
                            other_ins_coords = other_str.view(all_ins_coords.dtype).reshape(-1, all_ins_coords.shape[1])

                            # 2. 剔除遮挡
                            other_pixel_coordinates_in_view, other_z = projection(other_ins_coords, vol_origin, proj_mat, image_height, image_width)  # other instance的2D投影
                            
                            # 计算距离矩阵，并找到距离在固定像素以内的点
                            min_occluded_distance = 50
                            occluded_distances = np.sqrt(((other_pixel_coordinates_in_view[:, np.newaxis, :] - pixel_coordinates_in_view[np.newaxis, :, :]) ** 2).sum(axis=2))

                            close_points_mask = occluded_distances <= min_occluded_distance

                            z_diff_mask = other_z[:, np.newaxis] < cur_z

                            remove_mask = (close_points_mask & z_diff_mask).any(axis=0)

                            pixel_coordinates_in_view = pixel_coordinates_in_view[~remove_mask]

                            # # test
                            # imgae_copy = copy.deepcopy(image)
                            # coords_to_bwi(pixel_coordinates_in_view, imgae_copy, output_path="marked_bw_image_2.png")

                            """******************************************************************  直接用3D点来构造2D mask start ******************************************************************"""
                            if len(pixel_coordinates_in_view) > 50: 
                                "**************** SAM start ****************"
                                # 对点进行聚类滤波
                                n_clusters = 30  # 选择下采样后保留的点的数量

                                # 使用 K-means 聚类来进行下采样
                                kmeans = KMeans(n_clusters=n_clusters)
                                kmeans.fit(pixel_coordinates_in_view)

                                sam_input_point = kmeans.cluster_centers_  # 获取每个簇的中心点

                                # imgae_copy = copy.deepcopy(image)
                                # coords_to_bwi(sam_input_point, imgae_copy, output_path="marked_bw_image_3.png")

                                sam_image = image.convert("RGB")
                                sam_image = np.array(sam_image)
                                # sam_image = cv2.imread(img_path)
                                # sam_image = cv2.cvtColor(sam_image, cv2.COLOR_BGR2RGB)

                                # 预处理图像并输入SAM
                                sam_predictor.set_image(sam_image)

                                # 逐步添加点并更新掩膜
                                input_point = np.zeros((0, 2)) 
                                input_label = np.zeros((0))

                                # 存储初始掩膜
                                sam_masks = None
                                sam_masks_input = None

                                for sam_i in range(len(sam_input_point)):
                                    # 添加新点
                                    input_point = np.concatenate((input_point, [sam_input_point[sam_i]]), axis=0)
                                    input_label = np.concatenate((input_label, [1]), axis=0)  # 1表示positive point, 0表示negative point

                                    if sam_masks is None:  # 如果是第一次添加点，则直接进行推理
                                        sam_masks, scores, logits = sam_predictor.predict(
                                            point_coords=input_point,        # 输入坐标点
                                            point_labels=input_label,         # 输入点标签
                                            multimask_output=True             # 获取多个掩膜候选
                                        )
                                    elif sam_i != len(sam_input_point) - 1:   # 使用之前的掩膜与当前点一起进行推理
                                        sam_masks_input = logits[np.argmax(scores), :, :]  # 选择上一次最好的掩膜
                                        sam_masks, scores, logits = sam_predictor.predict(
                                            point_coords=input_point,         # 输入坐标点
                                            point_labels=input_label,         # 输入点标签
                                            mask_input=sam_masks_input[None, :, :],# 将上次的掩膜作为提示
                                            multimask_output=True            # 获取多个掩膜候选
                                        )
                                    else:  # 最后一次推理
                                        sam_masks_input = logits[np.argmax(scores), :, :]  # 选择上一次最好的掩膜
                                        sam_masks, scores, logits = sam_predictor.predict(
                                            point_coords=input_point,         # 输入坐标点
                                            point_labels=input_label,         # 输入点标签
                                            mask_input=sam_masks_input[None, :, :],# 将上次的掩膜作为提示
                                            multimask_output=False            # 输出单一掩膜
                                        )

                                sam_mask = sam_masks[0]

                                # # 腐蚀和膨胀操作以去除小的伪影
                                # sam_mask_uint8 = sam_mask.astype(np.uint8) * 255
                                # sam_kernel = np.ones((5, 5), np.uint8)
                                
                                # eroded_mask = cv2.erode(sam_mask_uint8, sam_kernel, iterations=1)  # 进行腐蚀操作（去除小零碎噪声区域）
                                # dilated_mask = cv2.dilate(eroded_mask, sam_kernel, iterations=1)  # 进行膨胀操作（填补小洞）
                                # sam_mask = dilated_mask.astype(np.bool_)  # 将腐蚀后的掩膜转换回布尔类型

                                # # 可视化
                                # alpha = 0.6
                                # sam_mask_uint8 = np.uint8(sam_mask * 255)
                                # sam_mask_colored = cv2.cvtColor(sam_mask_uint8, cv2.COLOR_GRAY2BGR)
                                # sam_mask_colored[sam_mask_uint8 == 255] = [255, 0, 0]  
                                # overlaid_sam_mask = cv2.addWeighted(np.array(image), 1 - alpha, sam_mask_colored, alpha, 0)
                                # overlaid_sam_mask = cv2.cvtColor(overlaid_sam_mask, cv2.COLOR_BGR2RGB)
                                # cv2.imwrite("overlaid_sam_mask.png", overlaid_sam_mask)

                                # print("Done.")

                                "**************** SAM end ****************"

                                "**************** 凸包 start ****************"
                                # mask_instance = np.zeros((image_height, image_width), dtype=np.uint8)

                                # # hull = ConvexHull(pixel_coordinates_in_view)  # 计算凸包(凸包（Convex Hull）是包含一组点的最小凸多边形)，将稀疏点连接成封闭区域 
                                # # hull_points = pixel_coordinates_in_view[hull.vertices]  # 按凸包顺序排列的点
                                # # hull_points = np.array(hull_points).reshape((-1, 1, 2)).astype(np.int32)  # 将凸包点转换为适合OpenCV的轮廓格式
                                # # cv2.fillPoly(mask_instance, [hull_points], 255)  # 在mask图像上绘制多边形

                                # tri = Delaunay(pixel_coordinates_in_view)  # 凸包画出的形状过于粗糙, Delaunay三角剖分生成更精细的封闭区域
                                # for simplex in tri.simplices:
                                #     triangle = pixel_coordinates_in_view[simplex].reshape((-1, 1, 2)).astype(np.int32)
                                #     cv2.fillPoly(mask_instance, [triangle], 255)

                                # # # 可视化
                                # # alpha = 0.0
                                # # mask_instance_colored = cv2.cvtColor(mask_instance, cv2.COLOR_GRAY2BGR)
                                # # mask_instance_colored[mask_instance == 255] = [255, 0, 0]  
                                # # overlaid_mask_instance = cv2.addWeighted(np.array(image), 1 - alpha, mask_instance_colored, alpha, 0)
                                # # overlaid_mask_instance = cv2.cvtColor(overlaid_mask_instance, cv2.COLOR_BGR2RGB)
                                # # cv2.imwrite("overlaid_mask_instance.png", overlaid_mask_instance)

                                # # print("Done.")
                                "**************** 凸包 end ****************"

                                """******************************************************************  直接用3D点来构造2D mask end ******************************************************************"""

                                """******************************************************************  利用2D mask来判断 start ******************************************************************"""
                                # # 获取图像中每个实例的mask，3D投影之后的像素坐标落在哪个实例mask中最多，则认为是那个实例
                                # ins_2d_unique = np.unique(ins_2d_img)  
                                # ins_2d_location_num, ins_2d_location_mask, mask_2d_class_all = [], [], []
                                # for ins_id in ins_2d_unique:
                                #     mask_2d = (ins_2d_img == ins_id)
                                #     mask_2d_class = cls_2d_img[mask_2d]  # mask的2d类别
                                #     mask_2d_class = np.bincount(mask_2d_class).argmax()
                                #     mask_2d_class_all.append(mask_2d_class)
                                #     ins_2d_location_mask.append(mask_2d)
                                    
                                #     rounded_coords = np.round(pixel_coordinates_in_view).astype(int)  
                                #     x_coords, y_coords = rounded_coords.T
                                #     x_coords = np.clip(x_coords, a_min=0, a_max=image_width-1) 
                                #     y_coords = np.clip(y_coords, a_min=0, a_max=image_height-1)
                                #     mask_hits = mask_2d[y_coords, x_coords]
                                #     ins_2d_location_num.append(np.sum(mask_hits))
                                
                                # # 为减少遮挡带来的实例选择出错，要求类别要对应
                                # sorted_indices_mask = sorted(range(len(ins_2d_location_num)), key=lambda i: ins_2d_location_num[i], reverse=True) 
                                # find_mask_flag = False
                                # for si_mask in sorted_indices_mask:
                                #     if mask_2d_class_all[si_mask] == 0:
                                #         continue
                                #     elif label_2d_mapping[str(mask_2d_class_all[si_mask])] not in class_name_dict:
                                #         continue
                                #     elif class_name_dict[label_2d_mapping[str(mask_2d_class_all[si_mask])]] == instance_data[candidate_ins_id]['semantic label']:
                                #         max_2d_mask = ins_2d_location_mask[si_mask]
                                #         find_mask_flag = True
                                #         break
                                #     else:
                                #         continue
                                # if find_mask_flag == False:
                                #     continue
                                """******************************************************************  利用2D mask来判断 end ******************************************************************"""

                                # cv2.imwrite('max_2d_mask.png', max_2d_mask * 255)
                                candidate_ins_2d_all.append({'class': instance_data[candidate_ins_id]['semantic label'], 
                                                            'mask': sam_mask.astype(bool),
                                                            'color': instance_data[candidate_ins_id]['color']})

                                candidate_count_dict[str(candidate_ins_id)] += 1

                    grounding_2d_data.append({
                        'instruction': instruction,
                        'img_path': img_path,
                        'candidate': candidate_ins_2d_all,
                    })

                # 确保场景中的每个instance至少被3张图片看到
                valid_scene_flag = True
                for candidate_ins_id in candidate_ins_ids:
                    if candidate_count_dict[str(candidate_ins_id)] < 3:
                        valid_scene_flag = False
                        break
                if  valid_scene_flag == True:               
                    with open(root_path + f'grounding_scene_rs_infos/{scene_name}_{instructions_i}_reasoning_seg_data.pkl', 'wb') as file:
                        pickle.dump(grounding_2d_data, file)
                        print("Done")


def generate_all_instance_mask():
    root_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/'
    root_path_orinal_data = '/root/paddlejob/workspace/env_run/zhouzhen05/Data/zhouzhen05/Data/'
    # root_path_2d_annotations = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_2d_annotations/'

    mode = 'train'
    scene_file = f'generate_color_label/scannetv2_{mode}.txt'
    # scene_file = 'generate_color_label/selected_scenes_1.txt'

    sam = sam_model_registry["vit_h"](checkpoint="generate_color_label/sam_vit_h_4b8939.pth")
    # sam = sam_model_registry["vit_l"](checkpoint="generate_color_label/sam_vit_l_0b3195.pth")
    sam.to(device="cuda:1")
    sam_predictor = SamPredictor(sam)

    print(scene_file, mode, sam.device)
    time.sleep(5)
    t_start = time.time()

    with open(scene_file, 'r') as split_file:
        for scene_name_i, scene_name in enumerate(split_file):
            print("**************************************************************** tqdm_scene: ", scene_name_i, "****************************************************************")
            scene_name = scene_name.strip()
            print("Processing scene_name: ", scene_name, "time: ", time.time() - t_start, "s.")

            if not scene_name in ['scene0385_02', 'scene0059_01']:
                continue

            # if os.path.exists(root_path + f'grounding_scene_mask_infos/{scene_name}_2d_mask_data.json.gz'):
            #     print(f'{scene_name}', 'exists')
            #     continue

            """ 获取所有实例的数据 start """
            rgb_path = root_path_orinal_data + 'all_tsdf_9_1/' + scene_name + '/full_rgb_layer0.npz'  # voxel_size: 0.04 
            rgb = np.load(rgb_path, allow_pickle=True)
            rgb = rgb.f.arr_0

            # 通过rgb去除黑色点来去除无效点
            black_index = rgb != np.array([0, 0, 0])
            all_ins_coords = np.argwhere(black_index[..., 0])
            """ 获取所有实例的数据 end """    
            with open(root_path + f'grounding_scene_instance_infos/{scene_name}_instance.pkl', 'rb') as file:
                instance_data = pickle.load(file)
            if len(instance_data) == 0:  # 1480比1513少了33，因为instance_data=0
                continue

            with open(f'/root/paddlejob/workspace/env_run/zhouzhen05/Data/zhouzhen05/Data/all_tsdf_9_1/fragments_{mode}.pkl', 'rb') as file:
                img_data_all = pickle.load(file)
            img_data = [item for item in img_data_all if item.get("scene") == scene_name]
            if len(img_data) == 0:
                continue
            all_image_ids = [image_id for item in img_data for image_id in item["image_ids"]]

            # 一个场景中所有的实例id
            candidate_ins_ids = np.arange(len(instance_data))

            mask_2d_data_per_scene = {}
            # ************************  staring process  *********************
            # 对于每张图像，将实例id映射到2D mask
            for img_id_count, img_id in enumerate(all_image_ids):
                print(img_id_count, "/", len(all_image_ids))
                candidate_mask_all_per_img = {}

                img_path = root_path_orinal_data + f'scans/{scene_name}/color/color_{img_id}.jpg'
                image = Image.open(img_path)
                image_width, image_height = image.size
            
                intrinsics = np.loadtxt(root_path_orinal_data + f'scans/{scene_name}/intrinsic/intrinsic_color.txt', delimiter=' ')[:3, :3]
                intrinsics = intrinsics.astype(np.float32)

                pose_path = root_path_orinal_data + f'scans/{scene_name}/pose/pose_{img_id}.txt'
                extrinsics = np.loadtxt(pose_path)

                vol_origin = img_data[0]['vol_origin']

                # Convert intrinsics and extrinsics matrices to a single projection matrix
                proj_mat = np.linalg.inv(extrinsics)
                proj_mat[:3, :4] = intrinsics @ proj_mat[:3, :4]

                for candidate_ins_id in candidate_ins_ids:  # 3D candidate
                    pixel_coordinates_in_view, cur_z = projection(instance_data[candidate_ins_id]['occ'], vol_origin, proj_mat, image_height, image_width)

                    if len(pixel_coordinates_in_view) > 50: 
                        """ 去除其他实例的遮挡 """
                        # 1. 剔除当前实例
                        current_ins_coords = instance_data[candidate_ins_id]['occ']
                        all_str = np.ascontiguousarray(all_ins_coords).view(np.dtype((np.void, all_ins_coords.dtype.itemsize * all_ins_coords.shape[1])))
                        cur_str = np.ascontiguousarray(current_ins_coords).view(np.dtype((np.void, current_ins_coords.dtype.itemsize * current_ins_coords.shape[1])))
                        other_str = np.setdiff1d(all_str, cur_str)  # 使用 np.setdiff1d 找出 all_str 中没有出现在 cur_str 中的位置
                        other_ins_coords = other_str.view(all_ins_coords.dtype).reshape(-1, all_ins_coords.shape[1])

                        # 2. 剔除遮挡
                        other_pixel_coordinates_in_view, other_z = projection(other_ins_coords, vol_origin, proj_mat, image_height, image_width)  # other instance的2D投影
                        
                        # 计算距离矩阵，并找到距离在固定像素以内的点
                        min_occluded_distance = 50
                        occluded_distances = np.sqrt(((other_pixel_coordinates_in_view[:, np.newaxis, :] - pixel_coordinates_in_view[np.newaxis, :, :]) ** 2).sum(axis=2))

                        close_points_mask = occluded_distances <= min_occluded_distance

                        z_diff_mask = other_z[:, np.newaxis] < cur_z

                        remove_mask = (close_points_mask & z_diff_mask).any(axis=0)

                        pixel_coordinates_in_view = pixel_coordinates_in_view[~remove_mask]

                        # # test
                        # imgae_copy = copy.deepcopy(image)
                        # coords_to_bwi(pixel_coordinates_in_view, imgae_copy, output_path="marked_bw_image_2.png")

                        """******************************************************************  直接用3D点来构造2D mask start ******************************************************************"""
                        if len(pixel_coordinates_in_view) > 50: 
                            "**************** SAM start ****************"
                            # 对点进行聚类滤波
                            n_clusters = 20  # 选择下采样后保留的点的数量

                            # 使用 K-means 聚类来进行下采样
                            kmeans = KMeans(n_clusters=n_clusters)
                            kmeans.fit(pixel_coordinates_in_view)

                            sam_input_point = kmeans.cluster_centers_  # 获取每个簇的中心点

                            # imgae_copy = copy.deepcopy(image)
                            # coords_to_bwi(sam_input_point, imgae_copy, output_path="marked_bw_image_3.png")

                            sam_image = image.convert("RGB")
                            sam_image = np.array(sam_image)
                            # sam_image = cv2.imread(img_path)
                            # sam_image = cv2.cvtColor(sam_image, cv2.COLOR_BGR2RGB)

                            # 预处理图像并输入SAM
                            sam_predictor.set_image(sam_image)

                            # 逐步添加点并更新掩膜
                            input_point = np.zeros((0, 2)) 
                            input_label = np.zeros((0))

                            # 存储初始掩膜
                            sam_masks = None
                            sam_masks_input = None

                            for sam_i in range(len(sam_input_point)):
                                # 添加新点
                                input_point = np.concatenate((input_point, [sam_input_point[sam_i]]), axis=0)
                                input_label = np.concatenate((input_label, [1]), axis=0)  # 1表示positive point, 0表示negative point

                                if sam_masks is None:  # 如果是第一次添加点，则直接进行推理
                                    sam_masks, scores, logits = sam_predictor.predict(
                                        point_coords=input_point,        # 输入坐标点
                                        point_labels=input_label,         # 输入点标签
                                        multimask_output=True             # 获取多个掩膜候选
                                    )
                                elif sam_i != len(sam_input_point) - 1:   # 使用之前的掩膜与当前点一起进行推理
                                    sam_masks_input = logits[np.argmax(scores), :, :]  # 选择上一次最好的掩膜
                                    sam_masks, scores, logits = sam_predictor.predict(
                                        point_coords=input_point,         # 输入坐标点
                                        point_labels=input_label,         # 输入点标签
                                        mask_input=sam_masks_input[None, :, :],# 将上次的掩膜作为提示
                                        multimask_output=True            # 获取多个掩膜候选
                                    )
                                else:  # 最后一次推理
                                    sam_masks_input = logits[np.argmax(scores), :, :]  # 选择上一次最好的掩膜
                                    sam_masks, scores, logits = sam_predictor.predict(
                                        point_coords=input_point,         # 输入坐标点
                                        point_labels=input_label,         # 输入点标签
                                        mask_input=sam_masks_input[None, :, :],# 将上次的掩膜作为提示
                                        multimask_output=False            # 输出单一掩膜
                                    )

                            sam_mask = sam_masks[0]

                            "**************** SAM end ****************"

                            compressed_mask = np.packbits(sam_mask.flatten()).tolist()  # 压缩
                            candidate_mask_all_per_img[f'{candidate_ins_id}'] = compressed_mask
                            
                mask_2d_data_per_scene[f'{img_id}'] = candidate_mask_all_per_img

            with gzip.open(root_path + f'grounding_scene_mask_infos/{scene_name}_2d_mask_data.json.gz', "wt", encoding="utf-8") as save_f:  # 压缩
                json.dump(mask_2d_data_per_scene, save_f)
                print('Done.')


def generate_VLM_grounding_data_from_mask():
    root_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/'
    root_path_orinal_data = '/root/paddlejob/workspace/env_run/zhouzhen05/Data/zhouzhen05/Data/'
    # root_path_2d_annotations = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_2d_annotations/'

    mode = 'train'  # base数据集mode都是train   
    scene_file = f'generate_color_label/scannetv2_train_4.txt'  

    print(scene_file, mode)
    time.sleep(5)
    t_start = time.time()

    with open(scene_file, 'r') as split_file:
        for scene_name_i, scene_name in enumerate(split_file):
            print("**************************************************************** tqdm_scene: ", scene_name_i, "****************************************************************")
            scene_name = scene_name.strip()
            print("Processing scene_name: ", scene_name, "time: ", time.time() - t_start, "s.")

            # if not scene_name in ['scene0343_00']:
            #         continue

            for instructions_i in range(instructions_num):
                print("instructions_i: ", instructions_i)

                if os.path.exists(root_path + f'grounding_scene_rs_infos_hard/{scene_name}_{instructions_i}_reasoning_seg_data.json.gz'):
                    print(f'{scene_name}_{instructions_i}', 'exists')
                    continue

                with open(f'/root/paddlejob/workspace/env_run/zhouzhen05/Data/zhouzhen05/Data/all_tsdf_9_1/fragments_{mode}.pkl', 'rb') as file:
                    img_data_all = pickle.load(file)
                img_data = [item for item in img_data_all if item.get("scene") == scene_name]
                if len(img_data) == 0:
                    continue
                all_image_ids = [image_id for item in img_data for image_id in item["image_ids"]]

                """ 获取所有实例的数据 start """
                rgb_path = root_path_orinal_data + 'all_tsdf_9_1/' + scene_name + '/full_rgb_layer0.npz'  # voxel_size: 0.04 
                rgb = np.load(rgb_path, allow_pickle=True)
                rgb = rgb.f.arr_0

                # 通过rgb去除黑色点来去除无效点
                black_index = rgb != np.array([0, 0, 0])
                all_ins_coords = np.argwhere(black_index[..., 0])
                """ 获取所有实例的数据 end """    
                with open(root_path + f'grounding_scene_instance_infos/{scene_name}_instance.pkl', 'rb') as file:
                    instance_data = pickle.load(file)
                if len(instance_data) == 0:
                    continue

                with open(root_path + f'grounding_scene_qa_infos_hard/{scene_name}_{instructions_i}_qa.pkl', 'rb') as file:
                    scene_qa = pickle.load(file)

                # 加载mask
                with gzip.open(root_path + f'grounding_scene_mask_infos/{scene_name}_2d_mask_data.json.gz', "rt", encoding="utf-8") as f:
                    compressed_mask_data = json.load(f)

                # save_instances_to_ply(instance_data, filename="combined_instances.ply", selected_inds=scene_qa[0]['instance ids'])

                # with open("generate_color_label/mapping.json", 'r', encoding='utf-8') as mapping_file:
                #     label_2d_mapping = json.load(mapping_file)

                class_name_dict = {3: 'cabinet', 4: 'bed', 5: 'chair', 
                    6: 'sofa', 7: 'table', 10: 'bookshelf', 
                    11: 'picture', 12: 'counter', 14: 'desk', 16: 'curtain', 24: 'refrigerator',
                    28: 'shower curtain', 33: 'toilet', 34: 'sink', 36: 'bathtub'}

                qa = scene_qa[0]  # 每个场景就一个qa
                print("qa: ", qa)
                instruction = qa['instruction']
                instance_ids = qa['instance ids']  # 3D answer
                candidate_ins_ids = np.array(qa['candidate'])  # 3D candidate
                grounding_2d_data = []

                candidate_count_dict = {}
                for candidate_ins_id in candidate_ins_ids:
                    candidate_count_dict[str(candidate_ins_id)] = 0

                # ************************  staring process  *********************
                # 对于每张图像，将实例id映射到2D mask
                for img_id_count, img_id in enumerate(all_image_ids):
                    print(img_id_count, "/", len(all_image_ids))
                    candidate_ins_2d_all = []
                    img_path = root_path_orinal_data + f'scans/{scene_name}/color/color_{img_id}.jpg'

                    for candidate_ins_id in candidate_ins_ids:  # 3D candidate
                        if str(candidate_ins_id) in compressed_mask_data[f'{img_id}'].keys():
                            # 解压缩数据
                            sam_mask = compressed_mask_data[f'{img_id}'][f'{candidate_ins_id}']
                            # sam_mask = np.unpackbits(np.array(sam_mask, dtype=np.uint8)).astype(bool)
                            # sam_mask = sam_mask.reshape(968, 1296)

                            candidate_ins_2d_all.append({'class': instance_data[candidate_ins_id]['semantic label'], 
                                                        'mask': sam_mask,
                                                        'color': instance_data[candidate_ins_id]['color']})
                                                        # TODO 'ins_id': candidate_ins_id

                            candidate_count_dict[str(candidate_ins_id)] += 1

                    # if len(candidate_ins_2d_all) < 2:  # 判断每张image中的instance个数
                    #     continue
                    grounding_2d_data.append({
                        'instruction': instruction,
                        'img_path': img_path,
                        'candidate': candidate_ins_2d_all,
                    })

                # 确保场景中的每个instance至少被3张图片看到
                valid_scene_flag = True
                for candidate_ins_id in candidate_ins_ids:
                    if candidate_count_dict[str(candidate_ins_id)] < 3:
                        valid_scene_flag = False
                        break
                if len(grounding_2d_data) == 0:
                    continue
                if  valid_scene_flag == True:               
                    # with open(root_path + f'grounding_scene_rs_infos/{scene_name}_{instructions_i}_reasoning_seg_data.pkl', 'wb') as file:
                    #     pickle.dump(grounding_2d_data, file)
                    #     print("Done")
                    with gzip.open(root_path + f'grounding_scene_rs_infos_hard/{scene_name}_{instructions_i}_reasoning_seg_data.json.gz', "wt", encoding="utf-8") as file:
                        json.dump(grounding_2d_data, file)
                        print("Save done")


def transfer_ins_to_fusion():
    root_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/grounding_scene_instance_infos'
    mapping_save_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/grounding_scene_instance_infos_mapping'

    ori_ins_files = [f for f in os.listdir(root_path) if f.endswith('.pkl')]

    for ori_ins_file in ori_ins_files:
        scene_name = ori_ins_file.split('_instance')[0]

        with open(os.path.join(root_path, ori_ins_file), 'rb') as file:
            ori_ins = pickle.load(file)

        rgb_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data/zhouzhen05/Data/all_tsdf_9_1/' + scene_name + '/full_rgb_layer0.npz'  # voxel_size: 0.04 
        rgb = np.load(rgb_path, allow_pickle=True)
        rgb = rgb.f.arr_0

        full_ins_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data/zhouzhen05/Data/all_tsdf_9_1/' + scene_name + '/full_instance_layer_interpolate0.npz'  # voxel_size: 0.04 
        full_ins = np.load(full_ins_path, allow_pickle=True)
        full_ins = full_ins.f.arr_0

        # 通过rgb去除黑色点来去除无效点
        black_index = rgb != np.array([0, 0, 0])
        coords = np.argwhere(black_index[..., 0])
        full_ins = full_ins[black_index[..., 0]]  # (N,)

        unique_ins_ids = np.unique(full_ins)
        full_dict = {}
        for ins_id in unique_ins_ids:
            mask = full_ins == ins_id
            full_dict[str(ins_id)] = coords[mask]

        mapping_dict = {}
        for i in range(len(ori_ins)):
            ious = []
            for value in full_dict.values():
                ious.append(compute_iou(ori_ins[i]['occ'], value))  # (N, 3)

            max_iou_index = ious.index(max(ious))
            key = list(full_dict.keys())[max_iou_index]
            mapping_dict[str(i)] = int(key)

        with open(os.path.join(mapping_save_path, f'{scene_name}.pkl'), 'wb') as file:
            pickle.dump(mapping_dict, file)
            print(os.path.join(mapping_save_path, f'{scene_name}.pkl'))
        


def compute_iou(A, B):
    # A: [N, 3], B: [M, 3]

    # 将 A 和 B 转换为集合形式，每个坐标点视为一个元组
    set_A = set(map(tuple, A))
    set_B = set(map(tuple, B))

    # 计算交集和并集
    intersection = set_A & set_B  # 交集
    union = set_A | set_B  # 并集

    # 计算 IoU
    iou = len(intersection) / len(union)

    return iou


def fuse_VLM_grounding_data():
    root_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/'
    mode = 'val'

    combined_list = []
    with open(f'generate_color_label/scannetv2_{mode}.txt', 'r') as split_file:
        for scene_name_i, scene_name in enumerate(split_file):
            scene_name = scene_name.strip() 
            print("tqdm_scene: ", scene_name_i, "/ 1201")

            if not os.path.exists(root_path + f'grounding_scene_rs_infos/{scene_name}_reasoning_seg_data.pkl'):
                print(scene_name, 'not exists')
                continue

            with open(root_path + f'grounding_scene_rs_infos/{scene_name}_reasoning_seg_data.pkl', 'rb') as file:
                grounding_2d_data = pickle.load(file)
            
            combined_list.extend(grounding_2d_data)

    print("reasoning seg data num: ", len(combined_list))
    with open(root_path + f'grounding_scene_rs_infos/{mode}_reasoning_seg_data.pkl', 'wb') as file:
        pickle.dump(combined_list, file)

    print("Done")


def generate_coco_format():
    root_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Code/DRecon/datasets/scannet/'
    with open(root_path + 'grounding_data/scene0000_00_vlm_grounding_data.pkl', 'rb') as file:
        grounding_2d_data = pickle.load(file)

    generate_coco_instruction_data(grounding_2d_data)

    print("Done")


def instance_color_mapping():
    color_mappings = {
        "Mahogany": "Red", "Burgundy": "Red", "Reddish": "Red", "Maroon": "Red",
        "Olive": "Green", "Teal": "Green",
        "Tan": "Brown",
        "Cream": "Beige",
        "Dark Silver": "Silver",
        "Dark Blue": "Blue",
        "Peach": "Pink"
    }

    root_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/'

    # 去除相近的颜色 
    with open(f'generate_color_label/scannetv2_val.txt', 'r') as split_file:
        for scene_name_i, scene_name in enumerate(split_file):
            print("**************************************************************** tqdm_scene: ", scene_name_i, "****************************************************************")
            scene_name = scene_name.strip()

            with open(root_path + f'grounding_scene_instance_infos/{scene_name}_instance.pkl', 'rb') as file:
                instance_data = pickle.load(file)
            if len(instance_data) == 0:
                print("empty")
                continue

            for instance in instance_data:
                original_color = instance["color"]
                if original_color in color_mappings:
                    instance["color"] = color_mappings[original_color]

            with open(root_path + f'grounding_scene_instance_infos/{scene_name}_instance.pkl', 'wb') as file:
                pickle.dump(instance_data, file)

            print("Done.")


def get_scene_type():
    with open("/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/scenetypes.pkl", 'rb') as pkl_file:
        already_data = pickle.load(pkl_file)

    folder_path = "/root/paddlejob/workspace/env_run/zhouzhen05/Data/zhouzhen05/Data"

    txt_files_dict = {}
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        print(filename)
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, filename)
        
        # 检查文件是否为txt文件
        if filename.endswith('.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    # 读取文件的所有行
                    lines = file.readlines()
                    # 获取最后一行（注意去掉换行符）
                    last_line = lines[-1].rstrip() if lines else ''
                    last_line = last_line.split(' = ')[-1]
                    # 将文件名和最后一行内容存入字典
                    txt_files_dict[filename] = last_line
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    with open("/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/scenetypes.pkl", 'wb') as pkl_file:
        pickle.dump(txt_files_dict, pkl_file)

def get_DRecon_scenes(): 
    with open('generate_color_label/scenetypes.pkl', 'rb') as f:
        scenetypes = pickle.load(f)

    # 预定义的类别列表
    categories = ['Bedroom / Hotel', 'Bathroom', 'Apartment']   # {'Bedroom / Hotel': 127, 'Bathroom': 70, 'Apartment': 11} 共 208 scans
    
    # 存储符合条件的scene_name的列表
    selected_scenes = []
    
    # 统计每个类别的数量
    category_counts = {category: 0 for category in categories}  
    
    with open('generate_color_label/scannetv2_train_sorted.txt', 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]

    seleted_scenes_count = 0
    for scene_name in lines:        
        # 在result_dict中查找对应的最后一行内容
        if scene_name + '.txt' in scenetypes:
            last_line_content = scenetypes[scene_name + '.txt']
            
            # 检查最后一行内容是否在预定义的类别列表中
            if last_line_content in categories:
                selected_scenes.append(scene_name)
                category_counts[last_line_content] += 1
                seleted_scenes_count += 1

                if seleted_scenes_count >= 125:
                    break
    
    # 将selected_scenes列表保存为一个新的Pickle文件
    output_path = 'generate_color_label/base_scenes.txt' 
    with open(output_path, "w", encoding="utf-8") as file:
        # 遍历列表，将每个元素写入文件的新行
        for item in selected_scenes:
            file.write(item + "\n")  # 在每个元素后面加上换行符

def select_train_val():
    # 读取 ori.txt 内容
    with open('generate_color_label/base_scenes.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 每隔5行抽取1行作为验证集
    val_lines = [lines[i] for i in range(4, len(lines), 5)]

    # 剩余的行作为训练集
    train_lines = [lines[i] for i in range(len(lines)) if i % 5 != 4]

    # 将 val_lines 写入 val.txt
    with open('generate_color_label/base_scenes_val.txt', 'w', encoding='utf-8') as val_file:
        val_file.writelines(val_lines)

    # 将 train_lines 写入 train.txt
    with open('generate_color_label/base_scenes_train.txt', 'w', encoding='utf-8') as train_file:
        train_file.writelines(train_lines)

    print("文件已分割完成, train.txt 和 val.txt 已生成。")


import os
import json
import re

def count_instance_per_image():
    # [SEG] > 1: 35200  8.5%  161605 - 35200 = 126405
    # [SEG] = 1: 377080  91.5%   
    # [SEG] = 0: 0  0.0%
    # all: 412280

    folder_path = "/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/reason_seg/Scannet_2D_Seg_full_83_pretrain/train"

    num = 0  
    file_count = 0
    for file_name in os.listdir(folder_path):
        # if file_name.endswith('.json.gz'): 
        if file_name.endswith('.json'):  
            file_path = os.path.join(folder_path, file_name)
            # print("file_count: ", file_count)
            file_count += 1

            # with gzip.open(file_path, "rt", encoding="utf-8") as file:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

                string = data['outputs']
                set_str = [match.strip() for match in re.findall(r'([\w\s]+)\s\[SEG\]', string)]
                len_SEG = string.count('SEG')
                if len_SEG > 1 and len(set(set_str)) == 1:
                    print(file_path)
                # for d in data:
                #     if len(d['candidate']) > 1:  # 每张图像的实例个数
                #         num += 1
                #         if file_name.split('_')[2] > 100:
                #             print(file_name)

            # if file_count % 100 == 0:
            #     print("file_count: ", file_count)
            #     print("current num: ", num)

    print("total num: ", num)


def split_file_by_lines():
    # 读取原始文件内容
    with open('generate_color_label/base_scenes_train.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 计算每个文件应包含的行数
    num_files = 4
    lines_per_file = len(lines) // num_files  # 每个文件25行

    # 将文件均匀分割并保存
    for i in range(num_files):
        start_idx = i * lines_per_file
        end_idx = start_idx + lines_per_file
        # 获取当前文件的25行
        split_lines = lines[start_idx:end_idx]
        
        # 写入新文件
        with open(f'generate_color_label/base_scenes_train_{i+1}.txt', 'w', encoding='utf-8') as out_file:
            out_file.writelines(split_lines)


if __name__ == "__main__":
    # generate_instance_data()  # 第 1 步 (跑一次即可)
    # instance_color_mapping()  # 去除相似的颜色
    get_scene_type()  # 获取场景类型
    # get_DRecon_scenes()  
    # select_train_val()
    # split_file_by_lines()  # 将第一个数据集的场景分成3份，加速计算

    # generate_grounding_3D_data()  # 第 2 步

    "原始数据格式保存"
    # generate_VLM_grounding_data()  # 第 3 步 (不推荐)
    "高效的数据存储格式"
    # generate_all_instance_mask()  # 第3.1步：将图片中的所有实例都事先分割出来 (跑一次即可)
    # generate_VLM_grounding_data_from_mask()  # 第3.2步：生成实例

    # fuse_VLM_grounding_data()  # 不需要
    # count_instance_per_image()  # 统计每张图像中的实例个数

    # transfer_ins_to_fusion()  # 第 4 步: 将reason_seg数据生产过程中，产生的实例id转换到3D实例fusion和refinement的id (跑一次即可)

    # generate_coco_format()

    print('Full Done')

