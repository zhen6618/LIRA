import numpy as np
from skimage import measure
import json
import shutil
import os
from PIL import Image
from skimage.transform import resize
import glob
import pickle
from collections import defaultdict

def merge_dict_values(dicts):
    result = defaultdict(list)
    for d in dicts:
        for key, value in d.items():
            result[key].append(value)

    ret = {}
    for key, values in result.items():
        ret[key] = values
    return ret


def mask_to_bbox(mask):
    """根据mask计算出bounding box"""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if rows.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return [int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)]
    else:
        return [0, 0, 0, 0]

def mask_to_polygon(mask):
    """根据mask计算出segmentation polygon"""
    contours = measure.find_contours(mask, 0.5)
    polygons = []

    # if len(contours) > 1:
    #     print("len(contours) > 1")
    #     image_array = (mask * 255).astype(np.uint8)
    #     image = Image.fromarray(image_array)
    #     image.save('/root/paddlejob/workspace/env_run/zhouzhen05/Code/DRecon/datasets/scannet/grounding_data/bool_image_pillow.png')

    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        if len(segmentation) >= 6:  # 至少有3个点才能构成一个有效的多边形
            polygons.append(segmentation)
    return polygons

def convert_to_coco(data):
    map_name_to_number = {
        'cabinet': 3, 'bed': 4, 'chair': 5, 'sofa': 6, 'table': 7, 
        'door': 8, 'window': 9, 'bookshelf': 10, 'picture': 11, 
        'counter': 12, 'desk': 14, 'curtain': 16, 'refrigerator': 24, 
        'shower curtain': 28, 'toilet': 33, 'sink': 34, 'bathtub': 36
    }

    coco_conversions = []
    coco_format = {
        "images": [],
        "annotations": []
    }
    annotation_id = 1

    root_path = "/root/paddlejob/workspace/env_run/zhouzhen05/Code/DRecon/datasets/scannet/"
    image_root_path = "/root/paddlejob/workspace/env_run/zhouzhen05/Code/DRecon/datasets/scannet/grounding_data/grounding_images/"
    image_map = {}
    # # 删除所有已有图片
    # png_files = glob.glob(image_root_path + '*.png')
    # list(map(os.remove, png_files))  # 使用map函数批量删除所有文件
    
    if os.path.exists(root_path + 'grounding_data/image_map.pkl'):
        with open(root_path + 'grounding_data/image_map.pkl', 'rb') as image_map_ff:
            image_map = pickle.load(image_map_ff)

    for item in data:
        # 存储图片，并判断image是否已经存在
        if item['img_path']+'original_img_path' in image_map:
            file_name = image_map[item['img_path']+'current_img_path']
            image_id = image_map[item['img_path']+'image_id']
        else:
            image_id = int(len(image_map)/3) + 1
            image_map[item['img_path']+'current_img_path'] = f'{int(image_id):012}.png'
            image_map[item['img_path']+'image_id'] = image_id
            image_map[item['img_path']+'original_img_path'] = item['img_path']

            current_img_path = image_root_path + image_map[item['img_path']+'current_img_path']
            shutil.copy(item['img_path'], current_img_path)  # 移动图片
            # resize image to 1024*1024
            with Image.open(current_img_path) as img:
                img = img.resize((1024, 1024))
                img.save(current_img_path)

            file_name = image_map[item['img_path']+'current_img_path']

            with open(root_path + 'grounding_data/image_map.pkl', 'wb') as image_map_f:
                pickle.dump(image_map, image_map_f)

        # 提取图像信息
        print(image_id)
        image_info = {
            "file_name": file_name,
            "id": image_id,
            "width": 1024,
            "height": 1024
        }
        coco_format['images'].append(image_info)

        # 如果mask存在且有实例
        color_class_ins_id = []
        if len(item['candidate']) > 0:  # 如果mask存在并且非空
            for candidate_i in range(len(item['candidate'])):
                mask = item['candidate'][candidate_i]['mask']  # 遍历每个mask
                # resize mask to 1024*1024
                mask = resize(mask.astype('float'), (1024, 1024), order=1, mode="constant", anti_aliasing=False)  # resize后变成浮点数
                mask = mask > 0.5

                bbox = mask_to_bbox(mask)
                segmentation = mask_to_polygon(mask)
                area = np.sum(mask)  # 面积为mask中True的数量

                mask_color = item['candidate'][candidate_i]['color']
                mask_class = item['candidate'][candidate_i]['class']

                color_class_ins_id.append({f'{mask_color} {mask_class}': annotation_id})

                annotation_info = {
                    "image_id": image_id,
                    "iscrowd": 0,  # 假设使用polygon存储
                    "category_id": map_name_to_number[mask_class],  # 这里假设所有mask对应同一类别
                    "area": int(area),
                    "segmentation": segmentation,
                    "bbox": bbox,
                    "id": annotation_id
                }
                coco_format['annotations'].append(annotation_info)
                annotation_id += 1

        # 存储对话信息
        conversations_single_info = {
            "id": f'{int(image_id):012}',
            "image": file_name,
        }

        conv_human = {
            'from': 'human',
            'value': item['instruction'] + '\n<image> (with grounding)',
        }

        if color_class_ins_id:  
            merge_ins = merge_dict_values(color_class_ins_id)

            conv_gpt['from'] = 'gpt'
            conv_gpt['value'] = ''
            conversations_single_info['gd_ls'] = []
            merge_count = 1
            for merge_k, merge_v in merge_ins.items():
                if merge_count < len(merge_ins):
                    conv_gpt['value'] = conv_gpt['value'] + '<g_s> ' + merge_k + ' <g_e> <seg>, '
                else:
                    conv_gpt['value'] = conv_gpt['value'] + '<g_s> ' + merge_k + ' <g_e> <seg>.'
                conversations_single_info['gd_ls'].append(merge_v)
                merge_count += 1
        else:
            conv_gpt = {
                'from': 'gpt',
                'value': 'no object.',
            }                 

        conversations_single_info['conversations'] = [conv_human, conv_gpt]
        coco_conversions.append(conversations_single_info)

        # 如果没有实例，不生成annotation

    with open(root_path + 'grounding_data/coco_conversions_scannet.json', 'w') as coco_f:
        json.dump(coco_conversions, coco_f)   

    coco_format['categories'] = [
        {"supercategory": "furniture", "id": 3, "name": "cabinet"},
        {"supercategory": "furniture", "id": 4, "name": "bed"},
        {"supercategory": "furniture", "id": 5, "name": "chair"},
        {"supercategory": "furniture", "id": 6, "name": "sofa"},
        {"supercategory": "furniture", "id": 7, "name": "table"},
        {"supercategory": "furniture", "id": 8, "name": "door"},
        {"supercategory": "furniture", "id": 9, "name": "window"},
        {"supercategory": "furniture", "id": 10, "name": "bookshelf"},
        {"supercategory": "furniture", "id": 11, "name": "picture"},
        {"supercategory": "furniture", "id": 12, "name": "counter"},
        {"supercategory": "furniture", "id": 14, "name": "desk"},
        {"supercategory": "furniture", "id": 16, "name": "curtain"},
        {"supercategory": "furniture", "id": 24, "name": "refrigerator"},
        {"supercategory": "furniture", "id": 28, "name": "shower curtain"},
        {"supercategory": "furniture", "id": 33, "name": "toilet"},
        {"supercategory": "furniture", "id": 34, "name": "sink"},
        {"supercategory": "furniture", "id": 36, "name": "bathtub"}
    ]

    with open(root_path + 'grounding_data/coco_format_scannet.json', 'w') as coco_f:
        json.dump(coco_format, coco_f)     

def generate_coco_instruction_data(grounding_2d_data):
    coco_data = convert_to_coco(grounding_2d_data)

