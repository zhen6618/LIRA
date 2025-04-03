
"*****************************************************   监控gpu利用率和显存占用   *****************************************************"
# import subprocess
# import time
# import re

# def monitor_gpu(interval, duration):
#     """
#     监控 GPU 使用情况，计算峰值显存占用和平均利用率。

#     :param interval: 数据采集间隔（秒）
#     :param duration: 监控总时长（秒）
#     :return: (峰值显存占用, 平均利用率)
#     """
#     memory_usage = []  # 显存占用列表
#     utilization = []  # GPU 利用率列表
#     end_time = time.time() + duration

#     while time.time() < end_time:
#         # 调用 nvidia-smi
#         result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", 
#                                  "--format=csv,nounits,noheader"], 
#                                 stdout=subprocess.PIPE, 
#                                 text=True)
        
#         # 解析输出
#         lines = result.stdout.strip().split('\n')
#         for line in lines:
#             memory, util = map(int, re.findall(r'\d+', line))
#             memory_usage.append(memory)
#             utilization.append(util)
        
#         # 等待下一个采样周期
#         time.sleep(interval)

#     # 计算结果
#     peak_memory = max(memory_usage) if memory_usage else 0
#     avg_utilization = sum(utilization) / len(utilization) if utilization else 0

#     return peak_memory, avg_utilization

# # 配置监控参数
# interval = 1  # 每隔 1 秒采集一次数据
# duration = 300  # 总共监控 600 秒
# print(f"开始监控 GPU 使用情况，每隔 {interval} 秒采集一次数据，总共监控 {duration} 秒。")

# # 开始监控
# peak_memory, avg_utilization = monitor_gpu(interval, duration)

# print(f"峰值显存占用: {peak_memory} MiB")  # 45GB
# print(f"平均 GPU 利用率: {avg_utilization:.2f} %")


"*****************************************************   统计qa中candidate的数量   *****************************************************"
# import os
# import pickle

# qa_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/grounding_scene_qa_infos_base/'
# letter_dict = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, 'l': 0}

# # 获取文件夹下所有的.pkl文件
# pkl_files = [f for f in os.listdir(qa_path) if f.endswith('.pkl')]

# # 遍历每个.pkl文件并读取内容
# for pkl_file in pkl_files:
#     file_path = os.path.join(qa_path, pkl_file)
#     with open(file_path, 'rb') as f:
#         data = pickle.load(f)
#         # cur_num = len(data[0]['instance ids'])
#         cur_num = len(data[0]['candidate'])

#         if cur_num > 9:
#             letter_dict['l'] += 1
#         else:
#             letter_dict[str(cur_num)] += 1

# print(letter_dict)

"*****************************************************   转换tsdf   *****************************************************"
# import numpy as np
# import os
# import open3d as o3d
# from skimage import measure

# path = '/dev/shm/all_tsdf_9_1/scene0000_00'

# tsdf_volume = np.load(os.path.join(path, 'full_tsdf_layer{}.npz'.format(0)), allow_pickle=True)
# tsdf_volume = tsdf_volume.f.arr_0

# tsdf_volume[tsdf_volume == 1] = np.nan  # single layer 设置为nan表示非法值，计算表面点云时会被忽略
# # tsdf_volume[tsdf_volume == 1] = np.inf  # double layer -1和inf之间会多一层

# # 使用Marching Cubes算法提取表面点云
# voxel_size = 0.04
# vertices, faces, normals, values = measure.marching_cubes(tsdf_volume, level=0.0)

# is_nan_flag = np.isnan(vertices).any(1)
# vertices_no_nan = vertices[~is_nan_flag]  # 去除nan值

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(vertices_no_nan * voxel_size)

# o3d.io.write_point_cloud('tsdf_to_mesh.ply', pcd)

# print("done")


"*****************************************************   open3d读取点云, 保存成numpy (OpenIns3D)    *****************************************************"
# import os
# import open3d as o3d
# import numpy as np
# import torch

# is_dir = True

# if is_dir:
#     # Define the folder containing your .ply files
#     folder_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/Others/OpenIns3D'  # Change this to your actual folder path

#     # List all files in the folder
#     ply_files = [f for f in os.listdir(folder_path) if f.endswith('.ply')]

#     # Iterate through each .ply file
#     for ply_file in ply_files:
#         ply_path = os.path.join(folder_path, ply_file)
        
#         # Read the point cloud from the ply file
#         pcd = o3d.io.read_point_cloud(ply_path)
        
#         # Get xyz and rgb data
#         xyz, rgb = np.asarray(pcd.points), np.asarray(pcd.colors) * 255.0
        
#         # Combine xyz and rgb into one array
#         xyz_rgb = torch.from_numpy(np.concatenate([xyz, rgb], axis=1)).float()
        
#         # Save as .npy with the same filename
#         save_path = os.path.join(folder_path, ply_file.replace('.ply', '.npy'))
#         np.save(save_path, xyz_rgb.numpy())

#         print(f"Saved: {save_path}")

# else:
#         ply_path = "/root/paddlejob/workspace/env_run/zhouzhen05/Code/Reason_Mapping/datasets/scannet/scans/scene0000_00/scene0000_00_vh_clean_2.ply"
        
#         # Read the point cloud from the ply file
#         pcd = o3d.io.read_point_cloud(ply_path)
        
#         # Get xyz and rgb data
#         xyz, rgb = np.asarray(pcd.points), np.asarray(pcd.colors) * 255.0
        
#         # Combine xyz and rgb into one array
#         xyz_rgb = torch.from_numpy(np.concatenate([xyz, rgb], axis=1)).float()
        
#         # Save as .npy with the same filename
#         save_path = ply_path.replace('.ply', '.npy')
#         np.save(save_path, xyz_rgb.numpy())

#         print(f"Saved: {save_path}")  


"*****************************************************   open3d读取点云, 保存成numpy (OpenMask3D)    *****************************************************"
# import os
# import open3d as o3d
# import numpy as np
# import pickle

# is_dir = True

# if is_dir:
#     # Define the folder containing your .ply files
#     folder_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/Others/OpenMask3D'  # Change this to your actual folder path

#     # List all files in the folder
#     ply_files = [f for f in os.listdir(folder_path) if f.endswith('.ply')]

#     # Iterate through each .ply file
#     for ply_file in ply_files:
#         ply_path = os.path.join(folder_path, ply_file)
        
#         pcd = o3d.io.read_point_cloud(ply_path)
#         pcd.estimate_normals()
#         coords = np.asarray(pcd.points)
#         colors = np.asarray(pcd.colors)
#         normals = np.asarray(pcd.normals)

#         data_dict = {'coords': coords, 'colors': colors, 'normals': normals}
        
#         pkl_path = ply_path.replace('.ply', '.pkl')
#         with open(pkl_path, 'wb') as f:
#             pickle.dump(data_dict, f)

#         print(f"Saved: {pkl_path}")

"*****************************************************   将某个文件复制到另外一个文件夹下    *****************************************************"


"*****************************************************   others    *****************************************************"
# import subprocess

# def run_curl():
#     # 定义curl命令
#     curl_command = [
#         "curl", "-o", "/dev/null", "-s", 
#         "-w", "DNS解析: %{time_namelookup}s\n连接建立: %{time_connect}s\n总时间: %{time_total}s\n", 
#         "https://chat.deepseek.com/"
#     ]
    
#     # 执行命令并获取输出
#     result = subprocess.run(curl_command, capture_output=True, text=True)
    
#     # 提取总时间
#     output = result.stdout
#     total_time = None
#     for line in output.split("\n"):
#         if "总时间:" in line:
#             # 去除 's' 字符并转换为 float
#             total_time_str = line.split(":")[1].strip().replace('s', '')
#             try:
#                 total_time = float(total_time_str)
#             except ValueError:
#                 print(f"无法转换总时间: {total_time_str}")
#                 continue
#             break
            
#     return total_time

# def main():
#     total_times = []
#     for _ in range(100):
#         total_time = run_curl()
#         if total_time is not None:
#             total_times.append(total_time)
    
#     if total_times:
#         average_time = sum(total_times) / len(total_times)
#         print(f"总时间的平均值: {average_time:.4f} 秒")  # 0.2648 秒
#     else:
#         print("没有获取到有效的总时间")

# if __name__ == "__main__":
#     main()


"*****************************************************   t-SNE可视化    *****************************************************"
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE

# np.random.seed(42)

# num_samples_per_category = 100

# instances = []
# features = []
# labels = []

# # 类 A：均值设为 0
# for i in range(1, num_samples_per_category + 1):
#     instances.append(f"A{i}")
#     feat = np.random.normal(loc=0.0, scale=1.0, size=128)
#     features.append(feat)
#     labels.append("A")

# # 类 B：均值设为 5
# for i in range(1, num_samples_per_category + 1):
#     instances.append(f"B{i}")
#     feat = np.random.normal(loc=5.0, scale=1.0, size=128)
#     features.append(feat)
#     labels.append("B")

# # 类 C：均值设为 10
# for i in range(1, num_samples_per_category + 1):
#     instances.append(f"C{i}")
#     feat = np.random.normal(loc=10.0, scale=1.0, size=128)
#     features.append(feat)
#     labels.append("C")

# features = np.array(features)
# print(features.shape)

# # 使用 t-SNE 将 128 维特征降维到 2D
# tsne = TSNE(n_components=2, random_state=42)
# features_2d = tsne.fit_transform(features)

# # 绘制 2D 散点图
# plt.figure(figsize=(10, 8))
# color_map = {'A': 'red', 'B': 'green', 'C': 'blue'}
# added_labels = set()  # 避免重复在图例中显示同一类别

# for i, inst in enumerate(instances):
#     clr = color_map[labels[i]]
#     lab = labels[i] if labels[i] not in added_labels else None
#     added_labels.add(labels[i])
#     plt.scatter(features_2d[i, 0], features_2d[i, 1], color=clr, label=lab, s=80)
#     # 在点旁边添加实例名称标注
#     # plt.text(features_2d[i, 0] + 0.3, features_2d[i, 1] + 0.3, inst, fontsize=8)

# # plt.xlabel("Dimension 1")
# # plt.ylabel("Dimension 2")
# plt.title("t-SNE Visualization of 128-D Instance Features")
# plt.legend()
# plt.grid(False)

# # 保存图片到文件
# plt.savefig("tsne_visualization.png", dpi=300)
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE

# # -------------------------------
# # 1. 构造示例数据
# # 假设你有两个实例 a 和 b，对应的向量我们这里构造成相同（实际中请替换成你的数据）
# np.random.seed(42)
# vector_a = np.random.randn(128)  # 实例 a 的向量
# vector_b = np.random.randn(128)  # 实例 b 的向量

# # 四个向量，分别属于 a, b, a, b
# vectors = np.array([vector_a, vector_b, vector_a*100, vector_b*100])
# # vectors = np.array([vector_a, vector_b, vector_a, vector_b])
# print(vectors[:, :10])
# norms = np.linalg.norm(vectors, axis=1, keepdims=True)
# vectors = vectors / norms  # 归一化

# vectors = np.tile(vectors, (50, 1))  # 复制 n_repeats 次

# print('')
# print(vectors[:4, :10])

# labels = ['a', 'b', 'a', 'b']

# # 如果你的向量保存在 PyTorch tensor 且在 GPU 上，请先调用 .cpu().numpy() 转为 numpy 数组

# # -------------------------------
# # 2. 使用 t-SNE 降维
# # 注意：perplexity 必须小于样本数量，这里样本数为4，所以 perplexity 可设为2
# tsne = TSNE(n_components=2)
# vectors_2d = tsne.fit_transform(vectors)

# # -------------------------------
# # 3. 可视化
# plt.figure(figsize=(6, 6))
# color_map = {'a': 'red', 'b': 'blue'}

# # 为了避免图例重复显示，记录已添加的标签
# legend_added = {}

# for i, (x, y) in enumerate(vectors_2d):
#     label = labels[i % 2]
#     clr = color_map[label]
#     # 只在第一次出现时添加图例标签
#     if label not in legend_added:
#         plt.scatter(x, y, color=clr, label=label, s=100)
#         legend_added[label] = True
#     else:
#         plt.scatter(x, y, color=clr, s=100)
#     # 添加点的标注，例如显示向量的索引
#     # plt.text(x + 0.1, y + 0.1, f'Vector {i}', fontsize=9)

# plt.xlabel('TSNE Component 1')
# plt.ylabel('TSNE Component 2')
# plt.title('t-SNE Visualization of 4 Vectors')
# plt.legend()
# plt.grid(True)

# # 如果需要保存图片，可调用 plt.savefig()（在 plt.show() 之前）
# plt.savefig('tsne_visualization.png', dpi=300)

# plt.show()


# import torch
# import numpy as np
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# # 输入数据
# vectors = torch.tensor([
#     [1.0, 0.0, 0.0, 0.0],  # 示例向量 1
#     [0.0, 1.0, 0.0, 0.0],  # 示例向量 2
#     [1.0, 0.0, 0.0, 0.0],  # 示例向量 3
#     [0.0, 1.0, 0.0, 0.0],   # 示例向量 4
#     [1.0, 0.0, 0.0, 0.0],  # 示例向量 1
#     [0.0, 1.0, 0.0, 0.0],  # 示例向量 2
#     [1.0, 0.0, 0.0, 0.0],  # 示例向量 3
#     [0.0, 1.0, 0.0, 0.0],   # 示例向量 4
#     [1.0, 0.0, 0.0, 0.0],  # 示例向量 1
#     [0.0, 1.0, 0.0, 0.0],  # 示例向量 2
#     [1.0, 0.0, 0.0, 0.0],  # 示例向量 3
#     [0.0, 1.0, 0.0, 0.0],   # 示例向量 4
#     [1.0, 0.0, 0.0, 0.0],  # 示例向量 1
#     [0.0, 1.0, 0.0, 0.0],  # 示例向量 2
#     [1.0, 0.0, 0.0, 0.0],  # 示例向量 3
#     [0.0, 1.0, 0.0, 0.0],   # 示例向量 4
#     [1.0, 0.0, 0.0, 0.0],  # 示例向量 1
#     [0.0, 1.0, 0.0, 0.0],  # 示例向量 2
#     [1.0, 0.0, 0.0, 0.0],  # 示例向量 3
#     [0.0, 1.0, 0.0, 0.0],   # 示例向量 4
#     [1.0, 0.0, 0.0, 0.0],  # 示例向量 1
#     [0.0, 1.0, 0.0, 0.0],  # 示例向量 2
#     [1.0, 0.0, 0.0, 0.0],  # 示例向量 3
#     [0.0, 1.0, 0.0, 0.0],   # 示例向量 4
#     [1.0, 0.0, 0.0, 0.0],  # 示例向量 1
#     [0.0, 1.0, 0.0, 0.0],  # 示例向量 2
#     [1.0, 0.0, 0.0, 0.0],  # 示例向量 3
#     [0.0, 1.0, 0.0, 0.0],   # 示例向量 4
#     [1.0, 0.0, 0.0, 0.0],  # 示例向量 1
#     [0.0, 1.0, 0.0, 0.0],  # 示例向量 2
#     [1.0, 0.0, 0.0, 0.0],  # 示例向量 3
#     [0.0, 1.0, 0.0, 0.0],   # 示例向量 4
# ], device='cuda:0')  # 替换为你的实际数据

# # 将向量 L2 归一化（确保长度相同）
# vectors_normalized = torch.nn.functional.normalize(vectors, p=2, dim=1)

# # 计算余弦相似度矩阵
# cosine_sim_matrix = torch.mm(vectors_normalized, vectors_normalized.T)

# # 将余弦相似度矩阵转换为距离矩阵（t-SNE 需要距离矩阵）
# distance_matrix = 1 - cosine_sim_matrix

# # 将距离矩阵转换为 numpy 数组（t-SNE 需要 numpy 输入）
# distance_matrix_np = distance_matrix.cpu().numpy()

# # 使用 t-SNE 降维
# tsne = TSNE(n_components=2, random_state=42)
# vectors_2d = tsne.fit_transform(distance_matrix_np)

# # 添加图例
# handles, labels_legend = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels_legend, handles))  # 去重
# plt.legend(by_label.values(), by_label.keys())

# # 保存图片
# plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')

# # 显示图片
# plt.show()

"*****************************************************   MDS可视化    *****************************************************"
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import MDS
# import seaborn as sns

# # 4x4 相似度矩阵
# similarity_matrix = np.array([
#     [1.0000, 0.0726, 1.0000, 0.0640],
#     [0.0726, 1.0000, 0.0760, 0.9997],
#     [1.0000, 0.0760, 1.0000, 0.0672],
#     [0.0640, 0.9997, 0.0672, 1.0000]
# ])

# # 将相似度转换为“距离”（这里用 1-相似度）
# distance_matrix = 1 - similarity_matrix

# # 生成 4 个实例的真实标签（随机分配不同类别）
# labels = np.array([0, 1, 2, 3])

# # 使用 MDS 进行降维
# mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
# embedding = mds.fit_transform(distance_matrix)

# # 绘制 MDS 结果
# plt.figure(figsize=(6, 4))
# sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, palette='viridis', s=200)
# plt.title("MDS Visualization of 4x4 Similarity Matrix")
# plt.xlabel("MDS Component 1")
# plt.ylabel("MDS Component 2")
# plt.legend(title="Labels")

# # 保存图片
# plt.savefig("mds_visualization.png")
# plt.show()



"*****************************************************   统计qa中candidate的数量   *****************************************************"
# import open3d as o3d
# import numpy as np

# def filter_red_mesh(input_ply, output_ply):
#     # 读取PLY文件
#     mesh = o3d.io.read_triangle_mesh(input_ply)
    
#     # 检查是否有颜色属性
#     if not mesh.has_vertex_colors():
#         print("该PLY文件没有颜色信息")
#         return
    
#     # 获取颜色数据
#     colors = np.asarray(mesh.vertex_colors)
    
#     # 定义目标颜色（在这里设置为你指定的颜色）
#     target_color = np.array([0.84705882, 0.74901961, 0.84705882])
    
#     # 使用np.isclose来比较颜色值，这样可以避免由于浮动精度问题导致的颜色匹配失败
#     color_diff = np.all(np.isclose(colors, target_color, atol=1e-6), axis=1)
    
#     # 筛选出符合条件的顶点
#     selected_vertices = np.asarray(mesh.vertices)[color_diff]  # 将mesh.vertices转换为numpy数组进行索引
#     selected_colors = colors[color_diff]
    
#     if len(selected_vertices) == 0:
#         print("没有找到匹配的颜色，返回空Mesh。")
#         return
    
#     # 创建一个新的mesh，只保留指定颜色的顶点
#     selected_mesh = o3d.geometry.TriangleMesh()
#     selected_mesh.vertices = o3d.utility.Vector3dVector(selected_vertices)
#     selected_mesh.vertex_colors = o3d.utility.Vector3dVector(selected_colors)
    
#     # 重新设置三角形面
#     selected_mesh.triangles = mesh.triangles
    
#     # 保存为新的PLY文件
#     o3d.io.write_triangle_mesh(output_ply, selected_mesh)
#     print(f"指定颜色的Mesh已保存到 {output_ply}")

# # 使用示例
# input_ply = "results/debug_global_fused_mesh_scene0009_01_0.ply"  # 输入的PLY文件
# output_ply = "results/debug_global_fused_mesh_scene0009_01_0_instance.ply"  # 输出的PLY文件

# filter_red_mesh(input_ply, output_ply)


"*****************************************************   查看scenetype   *****************************************************"
import pickle

with open('datasets/scenetypes.pkl', 'rb') as f:
    scenetypes = pickle.load(f)

print(scenetypes)

