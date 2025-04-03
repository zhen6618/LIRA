import os
import pickle
import numpy as np
import open3d as o3d
from skimage import measure
import plyfile
from openai import OpenAI
import trimesh
import time
import re
from copy import deepcopy

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.modules import evaluate_instance_segmentation
from tools.evaluation_utils import eval_depth, eval_mesh
from tools.map_metric_3d import MAPMetric3D
from tools.evaluate_semantic_instance import evaluate

def transform_mask(instance_infos):

    l = len(instance_infos)
    if l == 0:
        return None
    
    else:
        label = np.zeros_like(instance_infos[0]['mask']).astype(np.int64)
        for i in range(len(instance_infos)):
            label[instance_infos[i]['mask']] = i + 1

        return label

def save_ply(coords, rgb, labels, filename="point_cloud.ply"):
    # 确保坐标和标签数量一致
    assert len(coords) == len(labels), "Coordinates and labels must have the same length."
    
    # 定义20种颜色
    colors = [
        [0.9, 0.9, 0.9],   # 浅灰色 (label=0)
        [1.0, 0.0, 0.0],   # 红色
        [0.0, 1.0, 0.0],   # 绿色
        [0.0, 0.0, 1.0],   # 蓝色
        [1.0, 1.0, 0.0],   # 黄色
        [1.0, 0.0, 1.0],   # 紫色
        [0.0, 1.0, 1.0],   # 青色
        [0.5, 0.0, 0.0],   # 深红
        [0.0, 0.5, 0.0],   # 深绿
        [0.0, 0.0, 0.5],   # 深蓝
        [1.0, 0.5, 0.0],   # 橙色
        [0.5, 1.0, 0.0],   # 浅绿色
        [0.0, 1.0, 0.5],   # 浅青色
        [1.0, 0.0, 0.5],   # 粉色
        [0.5, 0.5, 1.0],   # 浅蓝
        [0.5, 1.0, 1.0],   # 浅紫
        [1.0, 1.0, 0.5],   # 浅黄
        [0.5, 0.0, 1.0],   # 深紫
        [0.0, 0.5, 1.0],   # 天蓝
    ]
    
    # 初始化点云数据
    num_points = len(coords)
    colors_assigned = rgb
    
    for i in range(num_points):
        label = labels[i]
        # label=0的点为灰色/原色
        if label == 0:
            # colors_assigned[i] = colors[0]
            continue
        else:
            # 标签大于0的，循环使用20种颜色
            colors_assigned[i] = colors[label % 20]
    
    # 转换为Open3D格式
    points = np.array(coords)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors_assigned)
    
    # 保存为PLY文件
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Point cloud saved to {filename}")


def save_single_mesh(vertices, faces, normals, values, mask, save_path):
    # 过滤掉无效点
    valid_indices = np.where(mask)[0]  # 获取保留的索引
    filtered_vertices = vertices[mask]
    filtered_normals = normals[mask]
    filtered_values = values[mask]

    # 重新映射 faces 索引
    index_map = np.full(vertices.shape[0], -1, dtype=int)  # 创建索引映射
    index_map[valid_indices] = np.arange(len(valid_indices))  # 仅保留有效点

    # 过滤 faces，去除包含无效点的三角形
    valid_faces_mask = np.all(mask[faces], axis=1)  # 仅保留所有点都有效的三角形
    filtered_faces = index_map[faces[valid_faces_mask]]  # 重新映射索引

    # 构建 trimesh 并导出
    mesh = trimesh.Trimesh(vertices=filtered_vertices, faces=filtered_faces, vertex_normals=filtered_normals)
    mesh.export(save_path)

    print(f"save to {save_path}")


def eval_geometry_reconstruction():
    pred_root_path = '/dev/shm/all_tsdf_9_1'

    scene_name_path = "datasets/base_scenes_vis.txt" 
    with open(scene_name_path, 'r') as f:
        scene_names = f.read().splitlines()

    dist1, dist2, prec, recal, fscore = [], [], [], [], []

    for scene_name in scene_names:
        print(scene_name, '/', len(scene_names))
        "*****************************   pred   *****************************"
        pred_path = f'{pred_root_path}/{scene_name}'

        tsdf_volume = np.load(os.path.join(pred_path, 'full_tsdf_layer{}.npz'.format(0)), allow_pickle=True)
        tsdf_volume = tsdf_volume.f.arr_0

        tsdf_volume[tsdf_volume == 1] = np.nan  # single layer 设置为nan表示非法值，计算表面点云时会被忽略
        # tsdf_volume[tsdf_volume == 1] = np.inf  # double layer -1和inf之间会多一层

        # 使用Marching Cubes算法提取表面点云
        voxel_size = 0.04
        vertices, faces, normals, values = measure.marching_cubes(tsdf_volume, level=0.0)

        is_nan_flag = np.isnan(vertices).any(1)
        vertices_no_nan = vertices[~is_nan_flag]  # 去除nan值

        save_single_mesh(vertices, faces, normals, values, ~is_nan_flag, f"single/single_{scene_name}.ply")

        with open(os.path.join(pred_root_path, 'fragments_train.pkl'), 'rb') as f:
            metas_train = pickle.load(f)
        with open(os.path.join(pred_root_path, 'fragments_val.pkl'), 'rb') as f:
            metas_val = pickle.load(f)
        metas = metas_train + metas_val

        selected_metas = [item for item in metas if item['scene'] in [scene_name]]
        vol_origin = selected_metas[0]['vol_origin']

        # save to ply
        pred_save_path = f'geometry_recon/pred_{scene_name}.ply'
        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(vertices_no_nan * voxel_size + vol_origin)
        o3d.io.write_point_cloud(pred_save_path, pcd_pred)

        "*****************************   gt   *****************************"
        gt_ply_path = "/root/paddlejob/workspace/env_run/zhouzhen05/Data/zhouzhen05/Data/ScanNetV2_ply/"

        gt_path = gt_ply_path + f'{scene_name}_vh_clean_2.ply'
        gt_plydata = plyfile.PlyData.read(gt_path)
        gt_coords = np.vstack([gt_plydata['vertex'][dim] for dim in ('x', 'y', 'z')]).T

        # save to ply
        gt_save_path = f'geometry_recon/gt_{scene_name}.ply'
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt_coords)
        o3d.io.write_point_cloud(gt_save_path, pcd_gt)

        "*****************************   eval   *****************************"
        metrics_mesh = eval_mesh(pred_save_path, gt_save_path)

        dist1.append(metrics_mesh["dist1"])
        dist2.append(metrics_mesh["dist2"])
        prec.append(metrics_mesh["prec"])
        recal.append(metrics_mesh["recal"])
        fscore.append(metrics_mesh["fscore"])

    dist1 = sum(dist1) / len(dist1)
    dist2 = sum(dist2) / len(dist2)
    prec = sum(prec) / len(prec)
    recal = sum(recal) / len(recal)
    fscore = sum(fscore) / len(fscore)

    print(f'dist1: {dist1}')  
    print(f'dist2: {dist2}') 
    print(f'prec: {prec}')
    print(f'recal: {recal}')
    print(f'fscore: {fscore}')


def eval_candidate_ins():
    folder_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Code/Reason_Mapping/scan_results_hard_val_composite'   
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    print("len: ", len(pkl_files))

    eval_path = 'evals_hard_val_composite'
    if not os.path.exists(eval_path):
        os.mkdir(eval_path)

    candidate_pred_ins = []
    candidate_gt_ins = []
    candidate_scene_names = []
    for pkl_file in pkl_files:
        candidate_scene_names.append(pkl_file.split('_scan_results')[0])

        file_path = os.path.join(folder_path, pkl_file)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
            candidate_pred_ins.append(data['candidate_pred_ins'])
            candidate_gt_ins.append(data['candidate_gt_ins']) 

    '*******************************************************************   eval   *******************************************************************'
    """
    preds = {
        "scene_name": {'pred_scores' = 100, 'pred_classes' = 100 'pred_masks' = Nx100},
        ...
    }
    gts: 
        scene_name.txt  (numpy ids)
        ...
    """
    CLASS_LABELS = ["cabinet", "bed", "chair", "sofa", "table", 
                    "bookshelf", "picture", "counter", "desk", "curtain", "refrigerator", 
                    "shower curtain", "toilet", "sink", "bathtub"]

    VALID_CLASS_IDS = np.array([3, 4, 5, 6, 7, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36])
    LABEL_TO_ID = {}
    for i in range(len(VALID_CLASS_IDS)):
        LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
    fake_len = 10000

    preds = {}
    for i in range(len(candidate_scene_names)):
        print(i, '/', len(candidate_scene_names), '/', candidate_scene_names[i])
        scene_name = candidate_scene_names[i]

        # 处理gt
        if len(candidate_gt_ins[i]) == 0:
            if len(candidate_pred_ins[i]) > 0:
                gt_instance = np.zeros(len(candidate_pred_ins[i][0]['mask']), dtype=np.int64)
                save_gt_path = eval_path + '/' + scene_name + '.txt'
                np.savetxt(save_gt_path, gt_instance, fmt='%d')
            else:
                gt_instance = np.zeros(fake_len, dtype=np.int64)
                save_gt_path = eval_path + '/' + scene_name + '.txt'
                np.savetxt(save_gt_path, gt_instance, fmt='%d')

        else:
            gt_instance = np.zeros(len(candidate_gt_ins[i][0]['mask']), dtype=np.int64)
            for j in range(len(candidate_gt_ins[i])):
                if candidate_gt_ins[i][j]['color_class'].split(' ')[1] == "shower":
                    class_id = 28
                else:
                    class_id = LABEL_TO_ID[candidate_gt_ins[i][j]['color_class'].split(' ')[1]]
                gt_instance[candidate_gt_ins[i][j]['mask']] = class_id * 1000 + j

            save_gt_path = eval_path + '/' + scene_name + '.txt'
            np.savetxt(save_gt_path, gt_instance, fmt='%d')

        # 处理preds
        if len(candidate_pred_ins[i]) == 0:
            if len(candidate_gt_ins[i]) > 0:
                preds[scene_name] = {'pred_scores': np.zeros(0), 'pred_classes': np.zeros(0), 'pred_masks': np.zeros((len(candidate_gt_ins[i][0]['mask']), 0))}
            else:
                preds[scene_name] = {'pred_scores': np.zeros(0), 'pred_classes': np.zeros(0), 'pred_masks': np.zeros((fake_len, 0))} 
        else:
            cur_candidate = candidate_pred_ins[i]

            pred_scores = []
            pred_classes = []
            pred_masks = []
            for j in range(len(cur_candidate)):
                pred_scores.append(1)

                pred_label = cur_candidate[j]['color_class'].split(' ')[1]  # scene0329_00_2
                if pred_label == "shower":
                    class_id = 28
                else:
                    class_id = LABEL_TO_ID[pred_label]
                pred_classes.append(class_id)
                pred_masks.append((cur_candidate[j]['mask'] * class_id).reshape(-1, 1))

            pred_scores = np.array(pred_scores)
            pred_classes = np.array(pred_classes)
            pred_masks = np.concatenate(pred_masks, axis=1)

            preds[scene_name] = {'pred_scores': pred_scores, 'pred_classes': pred_classes, 'pred_masks': pred_masks}


    aps = evaluate(preds, eval_path, f"{eval_path}/AP_3d.txt")
    mAP, ap50, ap25 = aps['all_ap'], aps['all_ap_50%'], aps['all_ap_25%']

    # ap25, ap50, ap75, mAP = evaluate_instance_segmentation(candidate_pred_ins, candidate_gt_ins)

    print(f"AP25: {ap25:.3f}, AP50: {ap50:.3f}, mAP: {mAP:.3f}") 


def get_LLM_inference(client, prompt):

    '******************************************************************   chatgpt   ******************************************************************'
    # completion = client.chat.completions.create(
    #     model="gpt-3.5-turbo",  # gpt-4o, gpt-4o-mini
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {
    #             "role": "user",
    #             "content": prompt,
    #         }
    #     ]
    # )

    # response = completion.choices[0].message.content

    '******************************************************************   deepseek   ******************************************************************'
    # completion = client.chat.completions.create(
    #     model="deepseek-chat",  # deepseek-chat (DeepSeek-V3), deepseek-reasoner (DeepSeek-R1)
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant"},
    #         {"role": "user", "content": prompt},
    #     ],
    #     stream=False
    # )

    # response = completion.choices[0].message.content

    '******************************************************************   QWen   ******************************************************************'
    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models   https://help.aliyun.com/zh/model-studio/getting-started/models#ced16cb6cdfsy
        model="qwen2.5-32b-instruct", # qwen2.5-0.5b/1.5b/3b/7b/14b/32b/72b-instruct
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}],
        )

    response = completion.choices[0].message.content

    return response

def LLM_RM():
    candidate_folder_path = 'scan_results_val_7B'  
    LLM_folder_path = 'scan_results_val_LLM_7B_chatgpt_4o_mini/'

    '**************   chatgpt   **************'
    # client = OpenAI()  # export OPENAI_API_KEY="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"
    '**************   deepseek   **************'  
    """
    测试网络时间延迟的办法： 
        1. 测试调用api的时间 - 直接测试延迟的时间: curl -o /dev/null -s -w "DNS解析: %{time_namelookup}s\n连接建立: %{time_connect}s\n总时间: %{time_total}s\n" https://chat.deepseek.com  # 0.42s
    """
    # client = OpenAI(api_key="sk-0c1d23ea08ca4931817e7246959b99fc", base_url="https://api.deepseek.com") 
    '**************   Qwen   **************'
    # curl -o /dev/null -s -w "DNS解析: %{time_namelookup}s\n连接建立: %{time_connect}s\n总时间: %{time_total}s\n" https://bailian.console.aliyun.com/  # 0.15s
    client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")  # Qwen_api_key="sk-c1d8099e06d4481685f8d75cfa8f2acb"


    pkl_files = [f for f in os.listdir(candidate_folder_path) if f.endswith('.pkl')]
    print("len: ", len(pkl_files))
    count = 0
    
    time_count = []

    for pkl_file in pkl_files:
        print(count, '/', len(pkl_files))
        count += 1

        file_path = os.path.join(candidate_folder_path, pkl_file)
        save_path = LLM_folder_path + file_path.split(candidate_folder_path)[1]
        # if os.path.exists(save_path):
        #     continue

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        candidate_data = data['candidate_pred_ins']
        instruction = data['instruction']

        "*************************************************************   LLM推理: 候选实例 ➡️ 目标实例   *************************************************************"
        prompt = "The following are some objects and their attribute information: \n "
        for i, ins in enumerate(candidate_data):
            prompt = prompt + "{ID: " + str(i) + ", " + "Color and Class: " + f"{ins['color_class']}, " + "Position Coordinate (x, y, z) (meter): " + f"{ins['bbox'][0:3]}, " + "Size (meter^3): " + str(np.round(np.prod(ins['bbox'][3:6]), 3)) + "} \n "        
        if len(candidate_data) == 0:
            prompt = prompt + "There is no object. \n "

        prompt = prompt + "Here is now an instruction. '" + instruction.split(" Please give all objects that are helpful in inferring final targets")[0] +  "' \n Find objects and their IDs that match the instruction based on the above object information. If it involves calculating the distance between objects, please make the judgment based on the Euclidean distance between the position coordinates of the objects. "         
        prompt = prompt + "When it comes to comparisons between objects, if there is only one object, no comparison is needed. The final result is strictly in the format of <0, 1, 2, ...>, strictly end with '<>' symbol. '<>' contains object IDs, and the ID inside should be in numeric form. If there is no object in the final result, return <>. "   
        # prompt = prompt + "Let's think step by step."
        prompt = prompt + "Let's respond briefly and quickly."  # "Let's think and respond briefly and quickly." / "Let's respond briefly and quickly." / "Let us think and respond briefly, quickly and accurately."
        # prompt = prompt + " Finally, write a brief sentence to summarize."
        # print("****************************************************************   prompt   ****************************************************************")
        print("prompt: ", prompt)

        t_infer_start = time.time()
        response = get_LLM_inference(client, prompt)
        t_infer_end = time.time()
        print("LLM inference time: ", t_infer_end - t_infer_start)
        time_count.append(t_infer_end - t_infer_start)
        if len(time_count) % 100 == 0 and len(time_count) > 0:
            print("avg time: ", sum(time_count) / len(time_count), '*' * 200)
        # print("****************************************************************   response   ****************************************************************")
        print("response: ", response)
        # print("*********************************************************************************************************************************************")

        # 解码
        # 寻找输出实例ID 定义正则表达式，匹配形式 {ID1, ID2, ID3} 的子串
        pattern = r'<([^>]+)>'  # 匹配 {} 内的内容
        matches = list(re.finditer(pattern, response))  # 从字符串的末尾开始查找符合条件的子串, 反转字符串进行查找
        if not matches:
            matched_ids = []
        else:
            # 获取最后一个匹配的内容
            last_match = matches[-1].group(1)  # group(1) 是匹配的第一个括号内的内容
            matched_ids = [id.strip() for id in last_match.split(',') if id.strip()]  # 分割该匹配为ID并验证其格式

        # 选择实例
        final_pred_ins = []
        for matched_id in matched_ids:
            if matched_id.isdigit():
                if int(matched_id) >= 0 and int(matched_id) < len(candidate_data):
                    final_pred_ins.append(candidate_data[int(matched_id)])
        data['final_pred_ins'] = final_pred_ins

        # save
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

        print('finish ', file_path)

# def eval_final_ins_old():
#     folder_path = 'scan_results_val_LLM_7B_chatgpt_4o'  
#     pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
#     print("len: ", len(pkl_files))

#     final_pred_ins = []
#     final_gt_ins = []
#     for pkl_file in pkl_files:
#         file_path = os.path.join(folder_path, pkl_file)
#         with open(file_path, 'rb') as f:
#             data = pickle.load(f)
            
#             final_pred_ins.append(data['final_pred_ins'])
#             final_gt_ins.append(data['final_gt_ins'])

#     ap25, ap50, ap75, mAP = evaluate_instance_segmentation(final_pred_ins, final_gt_ins)
#     print(f"AP25: {ap25:.3f}, AP50: {ap50:.3f}, mAP: {mAP:.3f}") 

def eval_final_ins():
    folder_path = 'scan_results_val_LLM_7B_chatgpt_4o_mini'  
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    print("len: ", len(pkl_files))

    eval_path = 'evals_val_7B_final'
    if not os.path.exists(eval_path):
        os.mkdir(eval_path)

    final_pred_ins = []
    final_gt_ins = []
    final_scene_names = []
    for pkl_file in pkl_files:
        final_scene_names.append(pkl_file.split('_scan_results')[0])

        file_path = os.path.join(folder_path, pkl_file)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
            final_pred_ins.append(data['final_pred_ins'])
            final_gt_ins.append(data['final_gt_ins'])

    '*******************************************************************   eval   *******************************************************************'
    """
    preds = {
        "scene_name": {'pred_scores' = 100, 'pred_classes' = 100 'pred_masks' = Nx100},
        ...
    }
    gts: 
        scene_name.txt  (numpy ids)
        ...
    """
    CLASS_LABELS = ["cabinet", "bed", "chair", "sofa", "table", 
                    "bookshelf", "picture", "counter", "desk", "curtain", "refrigerator", 
                    "shower curtain", "toilet", "sink", "bathtub"]

    VALID_CLASS_IDS = np.array([3, 4, 5, 6, 7, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36])
    LABEL_TO_ID = {}
    for i in range(len(VALID_CLASS_IDS)):
        LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
    fake_len = 10000

    preds = {}
    for i in range(len(final_scene_names)):
        print(i, '/', len(final_scene_names))
        scene_name = final_scene_names[i]

        # 处理gt
        if len(final_gt_ins[i]) == 0:
            if len(final_pred_ins[i]) > 0:
                gt_instance = np.zeros(len(final_pred_ins[i][0]['mask']), dtype=np.int64)
                save_gt_path = eval_path + '/' + scene_name + '.txt'
                np.savetxt(save_gt_path, gt_instance, fmt='%d')
            else:
                gt_instance = np.zeros(fake_len, dtype=np.int64)
                save_gt_path = eval_path + '/' + scene_name + '.txt'
                np.savetxt(save_gt_path, gt_instance, fmt='%d')

        else:
            gt_instance = np.zeros(len(final_gt_ins[i][0]['mask']), dtype=np.int64)
            for j in range(len(final_gt_ins[i])):
                if final_gt_ins[i][j]['color_class'].split(' ')[1] == "shower":
                    class_id = 28
                else:
                    class_id = LABEL_TO_ID[final_gt_ins[i][j]['color_class'].split(' ')[1]]
                gt_instance[final_gt_ins[i][j]['mask']] = class_id * 1000 + j

            save_gt_path = eval_path + '/' + scene_name + '.txt'
            np.savetxt(save_gt_path, gt_instance, fmt='%d')

        # 处理preds
        if len(final_pred_ins[i]) == 0:
            if len(final_gt_ins[i]) > 0:
                preds[scene_name] = {'pred_scores': np.zeros(0), 'pred_classes': np.zeros(0), 'pred_masks': np.zeros((len(final_gt_ins[i][0]['mask']), 0))}
            else:
                preds[scene_name] = {'pred_scores': np.zeros(0), 'pred_classes': np.zeros(0), 'pred_masks': np.zeros((fake_len, 0))} 
        else:
            cur_final = final_pred_ins[i]

            pred_scores = []
            pred_classes = []
            pred_masks = []
            for j in range(len(cur_final)):
                pred_scores.append(1)

                pred_label = cur_final[j]['color_class'].split(' ')[1]
                if pred_label == "shower":
                    class_id = 28
                else:
                    class_id = LABEL_TO_ID[pred_label]
                pred_classes.append(class_id)
                pred_masks.append((cur_final[j]['mask'] * class_id).reshape(-1, 1))

            pred_scores = np.array(pred_scores)
            pred_classes = np.array(pred_classes)
            pred_masks = np.concatenate(pred_masks, axis=1)

            preds[scene_name] = {'pred_scores': pred_scores, 'pred_classes': pred_classes, 'pred_masks': pred_masks}


    aps = evaluate(preds, eval_path, f"{eval_path}/AP_3d.txt")
    mAP, ap50, ap25 = aps['all_ap'], aps['all_ap_50%'], aps['all_ap_25%']

    # ap25, ap50, ap75, mAP = evaluate_instance_segmentation(final_pred_ins, final_gt_ins)

    print(f"AP25: {ap25:.3f}, AP50: {ap50:.3f}, mAP: {mAP:.3f}") 

# def eval_final_ins_others_old():
#     folder_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/OVIR-3D/results_reason_mapping_val'  
#     gt_path = 'scan_results_val_remove_global_bbox_feats'

#     pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
#     print("len: ", len(pkl_files))

#     final_pred_ins = []
#     final_gt_ins = []
#     for pkl_file in pkl_files:
#         file_path = os.path.join(folder_path, pkl_file)
#         with open(file_path, 'rb') as f:
#             pred_data = pickle.load(f)

#         gt_file_path = os.path.join(gt_path, pkl_file)
#         if not os.path.exists(gt_file_path):
#             continue
#         with open(gt_file_path, 'rb') as f:
#             gt_data = pickle.load(f)
            
#         final_pred_ins.append(pred_data)
#         final_gt_ins.append(gt_data['final_gt_ins'])
 
#     ap25, ap50, ap75, mAP = evaluate_instance_segmentation(final_pred_ins, final_gt_ins)
#     print(f"AP25: {ap25:.3f}, AP50: {ap50:.3f}, mAP: {mAP:.3f}") 

def eval_final_ins_others():
    folder_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/OVIR-3D/results_reason_mapping_val_implict_hard'  
    gt_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/Reason_Mapping/scan_results_val_7B_hard'
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    print("len: ", len(pkl_files))

    eval_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/Reason_Mapping/evals_OVIR-3D'
    if not os.path.exists(eval_path):
        os.mkdir(eval_path)

    pred_ins = []
    gt_ins = []
    scene_names = []
    for pkl_file in pkl_files:
        scene_names.append(pkl_file.split('_scan_results')[0])

        file_path = os.path.join(folder_path, pkl_file)
        with open(file_path, 'rb') as f:
            pred_data = pickle.load(f)

        gt_file_path = os.path.join(gt_path, pkl_file)
        if not os.path.exists(gt_file_path):
            continue
        with open(gt_file_path, 'rb') as f:
            gt_data = pickle.load(f)
            
        pred_ins.append(pred_data)

        implicit = True
        if implicit:
            gt_ins.append(gt_data['final_gt_ins'])
        else:
            gt_ins.append(gt_data['candidate_gt_ins'])

    '*******************************************************************   eval   *******************************************************************'
    """
    preds = {
        "scene_name": {'pred_scores' = 100, 'pred_classes' = 100 'pred_masks' = Nx100},
        ...
    }
    gts: 
        scene_name.txt  (numpy ids)
        ...
    """
    CLASS_LABELS = ["cabinet", "bed", "chair", "sofa", "table", 
                    "bookshelf", "picture", "counter", "desk", "curtain", "refrigerator", 
                    "shower curtain", "toilet", "sink", "bathtub"]

    VALID_CLASS_IDS = np.array([3, 4, 5, 6, 7, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36])
    LABEL_TO_ID = {}
    for i in range(len(VALID_CLASS_IDS)):
        LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
    fake_len = 10000

    preds = {}
    for i in range(len(scene_names)):
        print(i, '/', len(scene_names))
        scene_name = scene_names[i]

        # 处理gt
        if len(gt_ins[i]) == 0:
            if len(pred_ins[i]) > 0:
                gt_instance = np.zeros(len(pred_ins[i][0]['mask']), dtype=np.int64)
                save_gt_path = eval_path + '/' + scene_name + '.txt'
                np.savetxt(save_gt_path, gt_instance, fmt='%d')
            else:
                gt_instance = np.zeros(fake_len, dtype=np.int64)
                save_gt_path = eval_path + '/' + scene_name + '.txt'
                np.savetxt(save_gt_path, gt_instance, fmt='%d')

        else:
            gt_instance = np.zeros(len(gt_ins[i][0]['mask']), dtype=np.int64)
            for j in range(len(gt_ins[i])):
                if gt_ins[i][j]['color_class'].split(' ')[1] == "shower":
                    class_id = 28
                else:
                    class_id = LABEL_TO_ID[gt_ins[i][j]['color_class'].split(' ')[1]]
                gt_instance[gt_ins[i][j]['mask']] = class_id * 1000 + j

            save_gt_path = eval_path + '/' + scene_name + '.txt'
            np.savetxt(save_gt_path, gt_instance, fmt='%d')

        # 处理preds
        if len(pred_ins[i]) == 0:
            if len(gt_ins[i]) > 0:
                preds[scene_name] = {'pred_scores': np.zeros(0), 'pred_classes': np.zeros(0), 'pred_masks': np.zeros((len(gt_ins[i][0]['mask']), 0))}
            else:
                preds[scene_name] = {'pred_scores': np.zeros(0), 'pred_classes': np.zeros(0), 'pred_masks': np.zeros((fake_len, 0))} 
        else:
            cur = pred_ins[i]

            pred_scores = []
            pred_classes = []
            pred_masks = []
            for j in range(len(cur)):
                pred_scores.append(1)

                class_id = LABEL_TO_ID[cur[j]['color_class']]
                pred_classes.append(class_id)
                pred_masks.append((cur[j]['mask'] * class_id).reshape(-1, 1))

            pred_scores = np.array(pred_scores)
            pred_classes = np.array(pred_classes)
            pred_masks = np.concatenate(pred_masks, axis=1)

            preds[scene_name] = {'pred_scores': pred_scores, 'pred_classes': pred_classes, 'pred_masks': pred_masks}

    aps = evaluate(preds, eval_path, f"{eval_path}/AP_3d.txt")
    mAP, ap50, ap25 = aps['all_ap'], aps['all_ap_50%'], aps['all_ap_25%']

    print(f"AP25: {ap25:.3f}, AP50: {ap50:.3f}, mAP: {mAP:.3f}") 


def replace_rgb_in_ply(ply_file, labels, save_path):

    # 定义20种颜色
    defined_colors = [
        [0.9, 0.9, 0.9],   # 浅灰色 (label=0)
        [1.0, 0.0, 0.0],   # 红色
        [0.0, 1.0, 0.0],   # 绿色
        [0.0, 0.0, 1.0],   # 蓝色
        [1.0, 1.0, 0.0],   # 黄色
        [1.0, 0.0, 1.0],   # 紫色
        [0.0, 1.0, 1.0],   # 青色
        [0.5, 0.0, 0.0],   # 深红
        [0.0, 0.5, 0.0],   # 深绿
        [0.0, 0.0, 0.5],   # 深蓝
        [1.0, 0.5, 0.0],   # 橙色
        [0.5, 1.0, 0.0],   # 浅绿色
        [0.0, 1.0, 0.5],   # 浅青色
        [1.0, 0.0, 0.5],   # 粉色
        [0.5, 0.5, 1.0],   # 浅蓝
        [0.5, 1.0, 1.0],   # 浅紫
        [1.0, 1.0, 0.5],   # 浅黄
        [0.5, 0.0, 1.0],   # 深紫
        [0.0, 0.5, 1.0],   # 天蓝
    ]

    mesh = o3d.io.read_triangle_mesh(ply_file)

    # 获取顶点数据
    vertices = np.asarray(mesh.vertices)
    n_vertices = len(vertices)

    # 获取顶点的颜色数据，如果没有颜色，则创建一个空的颜色数组
    colors = np.asarray(mesh.vertex_colors)
    if colors.shape[1] != 3:
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.zeros((n_vertices, 3)))

    # 使用for循环逐一替换RGB值
    for i in range(n_vertices):
        label = labels[i]
        if label == 0:
            continue
        else:
            # 标签大于0的，循环使用20种颜色
            r, g, b = defined_colors[label % 20]
            mesh.vertex_colors[i] = [r, g, b]

    # 保存修改后的PLY文件
    o3d.io.write_triangle_mesh(save_path, mesh)

    print(f"Point cloud saved to {save_path}")


def visualization_ply():
    folder_path = 'scan_results_val_LLM_7B_chatgpt_4o_mini'  
    ply_root_path = "/root/paddlejob/workspace/env_run/zhouzhen05/Data/zhouzhen05/Data/ScanNetV2_ply/"
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    print("len: ", len(pkl_files))

    with open('results/infos.txt', 'w') as file:  # 清空txt中的内容
        pass  
    
    count = 0
    for pkl_file in pkl_files:
        count += 1
        # if count >= 10:
        #     print("-" * 50, "more than ", count, "-" * 50)
        #     break

        # if pkl_file != 'scene0032_00_33_scan_results.pkl': 
        #     continue    
        # else:
        #     print(pkl_file)

        file_path = os.path.join(folder_path, pkl_file)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        '---  快速寻找目标  ---'
        gt_inses = data['final_gt_ins']
        # # 统计target数量
        # if len(gt_inses) != 1: 
        #     continue
        # # 统计类别个数
        # classes = []
        # for gt_ins in gt_inses:
        #     color_class = gt_ins['color_class']
        #     class_split = color_class.split(' ')
        #     cur_cls= "".join(class_split[1:])

        #     classes.append(cur_cls)
        
        # if len(set(classes)) < 2:
        #     continue

        scene_name = pkl_file[:12]

        # if scene_name == 'scene0209_00' or scene_name == 'scene0172_01' or scene_name == 'scene0092_02':
        #     continue

        coords = data['gt_coords']
        # RGB
        ply_path = ply_root_path + f'{scene_name}_vh_clean_2.ply'
        # gt_plydata = plyfile.PlyData.read(ply_path)
        # vertices = gt_plydata['vertex']
        # xyz = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        # rgb = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T
        # rgb = rgb / 255.0

        '*************************************************************   candidate_pred实例可视化   *************************************************************'
        # if len(data['candidate_pred_ins']) == 0:
        #     mask_labels = np.zeros_like(coords[:, 0]).astype(np.int64)
        # else:
        #     mask_labels = transform_mask(data['candidate_pred_ins'])
        # save_path = 'results/' + pkl_file.replace('.pkl', '_candidate_pred.ply')
        # # save_ply(coords, rgb, mask_labels, save_path)
        # replace_rgb_in_ply(ply_path, mask_labels, save_path)

        '*************************************************************   candidate_gt实例可视化   *************************************************************'
        # if len(data['candidate_gt_ins']) == 0:
        #     mask_labels = np.zeros_like(coords[:, 0]).astype(np.int64)
        # else:
        #     mask_labels = transform_mask(data['candidate_gt_ins'])
        # save_path = 'results/' + pkl_file.replace('.pkl', '_candidate_gt.ply')
        # replace_rgb_in_ply(ply_path, mask_labels, save_path)

        '*************************************************************   final_pred实例可视化   *************************************************************'
        # if len(data['final_pred_ins']) == 0:
        #     mask_labels = np.zeros_like(coords[:, 0]).astype(np.int64)
        # else:
        #     mask_labels = transform_mask(data['final_pred_ins'])
        # save_path = 'results/' + pkl_file.replace('.pkl', '_final_pred.ply')
        # replace_rgb_in_ply(ply_path, mask_labels, save_path)

        '*************************************************************   final_gt实例可视化   *************************************************************'
        if len(data['final_gt_ins']) == 0:
            mask_labels = np.zeros_like(coords[:, 0]).astype(np.int64)
        else:
            mask_labels = transform_mask(data['final_gt_ins'])
        save_path = 'results/' + pkl_file.replace('.pkl', '_final_gt.ply')
        replace_rgb_in_ply(ply_path, mask_labels, save_path)

        '*************************************************************   save instruction   *************************************************************'
        gap="***"
        str_list = []
        str_list.append(str(count))

        instruction = data['instruction'].split('Please give all objects that are helpful in inferring final targets')[0]
        str_list.append(instruction)

        scene_instruction_name = pkl_file.split('_scan_results')[0]
        str_list.append(scene_instruction_name)

        gt_inses = data['final_gt_ins']
        for gt_ins in gt_inses:
            str_list.append(gt_ins['color_class'])

        with open('results/infos.txt', "a") as file:
            # 将列表中的元素用空隙连接起来
            line = gap.join(str_list)
            # 将连接后的字符串写入文件
            file.write(line + "\n")


        print("*" * 50, file_path, "done.")


def visualization_ply_others():
    ply_root_path = "/root/paddlejob/workspace/env_run/zhouzhen05/Data/zhouzhen05/Data/ScanNetV2_ply/"

    # other_method_pkl_root_path = "/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/OVIR-3D/results_reason_mapping_val_implict/"
    # other_method_pkl_root_path = "/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/Others/OpenIns3D/"
    other_method_pkl_root_path = "/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/Others/OpenMask3D/"

    pkl_file = 'scene0032_00_33_scan_results.pkl'    
    other_pkl_file = other_method_pkl_root_path + pkl_file

    with open(other_pkl_file, 'rb') as f:
        data = pickle.load(f)

    scene_name = pkl_file[:12]
    # RGB
    ply_path = ply_root_path + f'{scene_name}_vh_clean_2.ply'

    '*************************************************************   实例可视化   *************************************************************'
    # save_path = 'results/' + pkl_file.replace('.pkl', '_OVIR-3D.ply')
    # save_path = 'results/' + pkl_file.replace('.pkl', '_OpenIns3D.ply')
    save_path = 'results/' + pkl_file.replace('.pkl', '_OpenMask3D.ply')

    if len(data) == 0:
        mesh = o3d.io.read_triangle_mesh(ply_path)
        o3d.io.write_triangle_mesh(save_path, mesh)
        print(f"Point cloud saved to {save_path}")

    else:
        data = data[:3]
        mask_labels = transform_mask(data)
        replace_rgb_in_ply(ply_path, mask_labels, save_path)


if __name__ == '__main__':
    os.environ['http_proxy'] = 'agent.baidu.com:8188'
    os.environ['https_proxy'] = 'agent.baidu.com:8188'

    eval_geometry_reconstruction()

    # eval_candidate_ins()

    # LLM_RM()
    # eval_final_ins()
    # eval_final_ins_others() 

    # vis point cloud
    # visualization_ply()
    # visualization_ply_others()
    
