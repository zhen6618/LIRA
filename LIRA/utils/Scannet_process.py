import pickle
import json
import cv2
import numpy as np
import shutil
from PIL import Image
import os
import time
import gzip

def mask_to_polygons(mask):
    # 找到mask中的轮廓，使用cv2.RETR_CCOMP来处理多层结构（多对象分割）
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    
    polygons_list = []

    for i, contour in enumerate(contours):
        # 只处理最外层的轮廓或嵌套的分割区域
        if hierarchy[0][i][3] == -1:
            # 将轮廓点直接转换为 COCO 格式，即平展成 [x1, y1, x2, y2, ..., xn, yn]
            points = contour.reshape(-1, 2).tolist()
            flattened_points = [coord for point in points for coord in point]  # 将点展平成单个列表
            
            # 添加到COCO格式的分割列表中
            polygons_list.append(flattened_points)

    return polygons_list

seg_string = "[SEG]"
# root_path = "dataset/reason_seg/Scannet_2D_Seg_full/"
root_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/reason_seg/Scannet_2D_Seg_full_83_pretrain/'
supplement_instruction = " Please give all objects that are helpful in inferring final targets, as well as the category and color of these objects."
root_rs_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/'

# 定义类和颜色的排序顺序
classes = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'bookshelf', 
        'picture', 'counter', 'desk', 'curtain', 'refrigerator', 
        'shower curtain', 'toilet', 'sink', 'bathtub']

colors = ['Green', 'Copper', 'Brown', 'Red', 'White', 'Gold', 'Blue', 'Orange', 'Purple', 'Gray', 'Beige', 'Pink', 'Yellow', 'Silver', 'Black']

# 创建类和颜色的排序字典，用于在排序时提供权重
class_order = {cls: i for i, cls in enumerate(classes)}
color_order = {color: i for i, color in enumerate(colors)}

instructions_num = 360  # 每个场景的instruction数量

# 定义计算mask中心的方法
def mask_center(mask):
    indices = np.argwhere(mask)
    center = indices.mean(axis=0)  # 计算中心点
    return tuple(center)

mode = 'val'
read_path = f'utils/scannetv2_{mode}.txt'

t_start = time.time()
print(read_path)
time.sleep(5)
    
with open(read_path, 'r') as split_file: 
    for scene_name_i, scene_name in enumerate(split_file):
        print("**************************************************************** tqdm_scene: ", scene_name_i, "****************************************************************")
        scene_name = scene_name.strip()
        print("Processing scene_name: ", scene_name, "i: ", scene_name_i, "time: ", time.time() - t_start, "s.")

        # if not scene_name in ['scene0538_00']:
        #     continue

        for instructions_i in range(100, 100 + instructions_num):
            # if instructions_i != 646:
            #     continue

            if not os.path.exists(root_rs_path + f'grounding_scene_rs_infos_83_pretrain/{scene_name}_{instructions_i}_reasoning_seg_data.json.gz'):
                print(scene_name, 'not exists')
                continue        

            with gzip.open(root_rs_path + f'grounding_scene_rs_infos_83_pretrain/{scene_name}_{instructions_i}_reasoning_seg_data.json.gz', "rt", encoding="utf-8") as file:
                data = json.load(file)

            # 转换成LISA格式
            for _, d in enumerate(data):
                json_path = str(instructions_i) + '_' + d['img_path'].replace("/", "_")[1:].replace("jpg", "json")
                if os.path.exists(root_path + f'{mode}/' + json_path):
                    print(json_path, 'exists')
                    continue

                j_data = {}
                
                candidate_len = len(d['candidate'])
                if candidate_len == 0:
                    j_data['outputs'] = "No object."
                else:
                    candidate_data = d['candidate']

                    for kk in range(candidate_len):
                        candidate_data[kk]['mask'] = np.unpackbits(np.array(candidate_data[kk]['mask'], dtype=np.uint8)).astype(bool)

                        if candidate_data[kk]['mask'].shape[0] != 968 * 1296:
                            candidate_data[kk]['mask'] = candidate_data[kk]['mask'].reshape(480, 640)  # 个别场景的mask是480x640
                        else:
                            candidate_data[kk]['mask'] = candidate_data[kk]['mask'].reshape(968, 1296)

                    # 按照 class 的顺序排序，接着对同类中的元素按照 color 的顺序排序
                    sorted_candidate_data = sorted(
                        candidate_data,
                        key=lambda x: (class_order.get(x['class'], float('inf')), color_order.get(x['color'], float('inf')), mask_center(x['mask']))
                    )

                    j_data['outputs'] = ', '.join(f"{item['color'].lower()} {item['class']} {seg_string}" for item in sorted_candidate_data) + "."

                j_data['text'] = [d['instruction'] + supplement_instruction]
                j_data['is_sentence'] = True
                j_data['image'] = d['img_path']

                j_data_shapes = {}
                j_data_shapes['label'] = 'target'
                j_data_shapes['labels'] = ['target']
                j_data_shapes['shape_type'] = "polygon"
                j_data_shapes['image_name'] = d['img_path']

                if candidate_len == 0:
                    j_data_shapes['points'] = None
                else:
                    points = []
                    for c in range(candidate_len):
                        points.append(mask_to_polygons(sorted_candidate_data[c]['mask']))  # 已经排好序
                    j_data_shapes['points'] = points

                j_data_shapes['flags'] = {}

                j_data['shapes'] = [j_data_shapes]

                with open(root_path + f'{mode}/' + json_path, "w") as json_file:
                    json.dump(j_data, json_file)  

                # image_re = Image.open(d['img_path'])
                # image_re = image_re.resize((1024, 1024))
                # image_re.save(root_path + 'train/' + all_info['image'])     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  记得在LISA中把图像resize成1024x1024 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                print(root_path + f'{mode}/' + json_path, 'done.')

print('Done.')
