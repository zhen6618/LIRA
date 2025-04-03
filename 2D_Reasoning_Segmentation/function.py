# import deepspeed
# import torch
# import os
# import socket

# # 配置端口
# master_port = 24999  # 24999

# # 尝试绑定指定端口，查看是否被占用
# def check_port_in_use(port):
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         try:
#             s.bind(('0.0.0.0', port))
#             return False  # 端口未被占用
#         except OSError:
#             return True  # 端口已被占用

# if check_port_in_use(master_port):
#     print(f"端口 {master_port} 已被占用")
# else:
#     print(f"端口 {master_port} 未被占用")

# if check_port_in_use(29500):
#     print(f"端口 {29500} 已被占用")
# else:
#     print(f"端口 {29500} 未被占用")

# if not check_port_in_use(master_port):
#     # 模拟一个简单的模型和训练配置
#     model = torch.nn.Linear(10, 10).cuda()

#     # DeepSpeed 配置
#     ds_config = {
#         "train_batch_size": 8,
#         "gradient_accumulation_steps": 2,
#         "fp16": {
#             "enabled": True
#         },
#         "zero_optimization": {
#             "stage": 2,
#             "contiguous_gradients": True,
#             "overlap_comm": True,
#             "reduce_scatter": True,
#             "reduce_bucket_size": 5e8,
#             "allgather_bucket_size": 5e8,
#         },
#     }

#     # 设置环境变量 MASTER_PORT
#     # os.environ['MASTER_PORT'] = str(master_port)
    
#     try:
#         # 尝试初始化 DeepSpeed
#         model_engine, optimizer, _, _ = deepspeed.initialize(
#             model=model,
#             config_params=ds_config,
#             dist_init_required=True,
#         )
#         print(f"DeepSpeed 成功初始化，使用端口 {master_port}")
#     except Exception as e:
#         print(f"初始化 DeepSpeed 时出错: {e}")






"""************************************************************************   json image path mapping   ***********************************************************************"""
# import os
# import json
# import pickle
# from tqdm import tqdm 

# def json_to_pkl(json_folder, output_pkl_path):
#     data_dict = {}
#     json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    
#     # 遍历文件夹中的所有文件
#     for filename in tqdm(json_files, desc="Processing JSON files"):
#         json_path = os.path.join(json_folder, filename)
        
#         # 打开并读取JSON文件内容
#         with open(json_path, 'r') as f:
#             json_data = json.load(f)
            
#             # 获取图像路径并转换为绝对路径
#             image_path = json_data.get('image')

#             image_path = image_path.replace('/root/paddlejob/workspace/env_run/zhouzhen05', '/home/users/zhouzhen05')
#             json_path = os.path.abspath(json_path)
#             json_path = json_path.replace('/root/paddlejob/workspace/env_run/zhouzhen05', '/home/users/zhouzhen05')
                
#             # 将JSON文件路径与图像路径存入字典
#             data_dict[json_path] = image_path

#     # 将字典保存为PKL文件
#     with open(output_pkl_path, 'wb') as pkl_file:
#         pickle.dump(data_dict, pkl_file)

# # 使用示例
# json_folder = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/reason_seg/Scannet_2D_Seg_full_only_mask/train'  # JSON文件所在的文件夹路径
# output_pkl_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/reason_seg/Scannet_2D_Seg_full_only_mask/train_mapping.pkl'  # 输出PKL文件路径
# json_to_pkl(json_folder, output_pkl_path)






"""************************************************************************   只选mask的数据   ***********************************************************************"""
import os
import json
import shutil
from tqdm import tqdm

# 源文件夹和目标文件夹路径
source_folder = "/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/reason_seg/Scannet_2D_Seg_full/val"  # 替换为你的json文件夹路径
destination_folder = "/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/reason_seg/Scannet_2D_Seg_full_only_mask/val"  # 替换为保存json文件的目标文件夹路径

# 确保目标文件夹存在
os.makedirs(destination_folder, exist_ok=True)

# 获取文件列表
files = [f for f in os.listdir(source_folder) if f.endswith(".json")]

# 使用tqdm显示进度条
for filename in tqdm(files, desc="Processing JSON files"):
    file_path = os.path.join(source_folder, filename)

    # 读取json文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 检查 'outputs' 键的值中是否包含 '[SEG]'
    if 'outputs' in data and '[SEG]' in str(data['outputs']):
        # 复制符合条件的json文件到目标文件夹
        shutil.copy(file_path, os.path.join(destination_folder, filename))

print("文件筛选和复制完成。")

