# import os
# import pickle

# with open('datasets/base_scenes_val.txt', 'r') as file:
#     lines = file.readlines()
# val_scene_names = [line.strip() for line in lines]

# # 指定文件夹路径
# qa_path = "/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/grounding_scene_qa_infos_base_new/"

# # 获取文件夹下所有.pkl文件
# pkl_files = [f for f in os.listdir(qa_path) if f.endswith('.pkl')]

# count = 0
# candidate_answer_count = []
# # 依次读取每个.pkl文件
# for pkl_file in pkl_files:
#     scene_name = pkl_file[:12]
#     # if scene_name not in val_scene_names:
#     if scene_name not in ['scene0006_01']:
#         continue

#     file_path = os.path.join(qa_path, pkl_file)
    
#     # 打开并加载.pkl文件
#     with open(file_path, 'rb') as file:
#         data = pickle.load(file)
    
#     candidate = data[0]['candidate']

#     instruction = data[0]['instruction']
#     print('count: ', count, ', instruction: ', instruction)
#     count += 1

#     if len(candidate) == 1:
#         candidate_answer_count.append(True)
#     else:
#         candidate_answer_count.append(False)

# print("")

import shutil
import os

def copy_files(file_list, src_folder, dest_folder):
    # 确保目标文件夹存在
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for file in file_list:
        # 构造源文件和目标文件的完整路径
        src_file = os.path.join(src_folder, file)
        dest_file = os.path.join(dest_folder, file)
        
        # 检查源文件是否存在
        if os.path.exists(src_file):
            shutil.copy(src_file, dest_file)
            print(f"文件 {file} 已复制到 {dest_folder}")
        else:
            print(f"文件 {file} 在源文件夹中不存在")

# 示例用法
file_list = [
    'scene0006_01_0_scan_results.pkl',
    'scene0006_01_3_scan_results.pkl',
    'scene0006_01_4_scan_results.pkl',
    'scene0006_01_5_scan_results.pkl',
    'scene0006_01_9_scan_results.pkl',
    'scene0006_01_13_scan_results.pkl',
    'scene0006_01_16_scan_results.pkl',
    'scene0006_01_21_scan_results.pkl',
    'scene0006_01_23_scan_results.pkl',
    'scene0006_01_24_scan_results.pkl',
    'scene0006_01_26_scan_results.pkl',
    'scene0006_01_32_scan_results.pkl',
    'scene0006_01_34_scan_results.pkl',
    'scene0006_01_37_scan_results.pkl',
    'scene0006_01_39_scan_results.pkl',
    'scene0006_01_40_scan_results.pkl',
    'scene0006_01_47_scan_results.pkl',
    'scene0006_01_49_scan_results.pkl',
    'scene0006_01_52_scan_results.pkl',
    'scene0006_01_57_scan_results.pkl',
    'scene0006_01_63_scan_results.pkl',
    'scene0006_01_65_scan_results.pkl',
    'scene0006_01_68_scan_results.pkl',
    'scene0006_01_72_scan_results.pkl',
    'scene0006_01_74_scan_results.pkl',
    'scene0006_01_75_scan_results.pkl',
    'scene0006_01_77_scan_results.pkl',
    'scene0006_01_78_scan_results.pkl',
    'scene0006_01_80_scan_results.pkl',
    'scene0006_01_81_scan_results.pkl',
    'scene0006_01_84_scan_results.pkl',
    'scene0006_01_89_scan_results.pkl',
    'scene0006_01_96_scan_results.pkl',
    'scene0006_01_98_scan_results.pkl',
    'scene0006_01_99_scan_results.pkl',
    ]  # 文件名列表
src_folder = 'scan_results_val_7B_epoch_4'  # 源文件夹路径
dest_folder = 'scan_results_val_ovir'  # 目标文件夹路径

copy_files(file_list, src_folder, dest_folder)

