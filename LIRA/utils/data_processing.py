import glob
import json
import os
import time
import cv2
import numpy as np
from skimage.transform import resize


def get_mask_from_json(json_path, img, original_size):
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    inform = anno["shapes"]  
    comments = anno["text"]
    is_sentence = anno["is_sentence"]
    answer = anno["outputs"]

    height, width = img.shape[:2]

    ### sort polies by area
    # area_list = []
    # valid_poly_list = []
    for i in inform: 
        label_id = i["label"]
        points = i["points"]
        if "flag" == label_id.lower():  ## meaningless deprecated annotations
            continue

        if points == None:
            masks = None
            continue
        
        masks = []
        for pot in points:  # 有几个pot，就有几个instance
            mask = np.zeros((original_size[0], original_size[1]), dtype=np.uint8)
            for polygon in pot:
                polygon = np.array(polygon).reshape(-1, 2)  # 将平展的点转换为 (x, y) 的形状
                cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)  # 1 表示前景  在mask上填充多边形区域
            mask = resize(mask.astype('float'), (1024, 1024), order=1, mode="constant", anti_aliasing=False)  # resize后变成浮点数
            mask[mask >= 0.5] = 1
            mask[mask < 0.5] = 0
            masks.append(mask)

        # # 按照从左上到右下的顺序，给mask排序
        # if len(masks) > 1:
        #     # 计算 mask 的中心点
        #     def get_center(mask):
        #         coords = np.argwhere(mask == 1)  # 找到 mask 中所有非零点的坐标
        #         center = coords.mean(axis=0)  # 计算中心点坐标
        #         return tuple(center)

        #     masks = sorted(masks, key=get_center)  # 根据中心点的 y, x 坐标排序，从左上到右下
        #     # # 查看排序后的结果
        #     # for m in masks:
        #     #     print(get_center(m))  # 输出中心点坐标，检查是否按顺序排列

        for mi in range(len(masks)):
            masks[mi] = np.expand_dims(masks[mi], axis=0)
        masks = np.concatenate(masks, axis=0)
    # import pdb;pdb.set_trace()
    return masks, comments, is_sentence, answer


if __name__ == "__main__":
    is_dir = False

    if is_dir:
        data_dir = "dataset/reason_seg/Scannet_2D_Seg_full/train"
        vis_dir = "dataset/reason_seg/Scannet_2D_Seg_full/vis"

        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        json_path_list = sorted(glob.glob(data_dir + "/*.json"))
        for json_path in json_path_list:
            with open(json_path, "r") as r:
                anno_json = json.loads(r.read())

            # img_path = json_path.replace(".json", ".png")
            img_path = anno_json['image']
            img = cv2.imread(img_path)[:, :, ::-1]
            original_size = img.shape[:2]
            img = cv2.resize(img, (1024, 1024))

            # In generated mask, value 1 denotes valid target region, and value 255 stands for region ignored during evaluaiton.
            mask, comments, is_sentence, answer = get_mask_from_json(json_path, img, original_size)

            if mask is not None:
                np.random.seed(0)  # For consistent color selection
                colors = np.random.randint(0, 255, size=(mask.shape[0], 3))
                vis_img = img.copy()

                for i in range(mask.shape[0]):
                    # visualization. Green for target, and red for ignore.
                    valid_mask = (mask[i] == 1).astype(np.float32)[:, :, None]
                    ignore_mask = (mask[i] == 255).astype(np.float32)[:, :, None]

                    # Apply colors for each instance based on the mask
                    color = colors[i]
                    vis_img = vis_img * (1 - valid_mask) * (1 - ignore_mask) + (
                        (color * 0.6 + img * 0.4) * valid_mask
                        + (np.array([255, 0, 0]) * 0.6 + img * 0.4) * ignore_mask
                    )

                vis_img = np.concatenate([img, vis_img], 1)
                
                vis_path = os.path.join(vis_dir, json_path.split("/")[-1].replace(".json", ".jpg"))
                cv2.imwrite(vis_path, vis_img[:, :, ::-1])
                print("Visualization has been saved to: ", vis_path)

    else:
        json_path = "/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/reason_seg/Scannet_2D_Seg_full_83_pretrain/train/83_root_paddlejob_workspace_env_run_zhouzhen05_Data_zhouzhen05_Data_scans_scene0656_03_color_color_29.json"
        vis_path = "chat_label_watch.png"

        with open(json_path, "r") as r:
            anno_json = json.loads(r.read())

        # img_path = json_path.replace(".json", ".png")
        img_path = anno_json['image']
        img = cv2.imread(img_path)[:, :, ::-1]
        original_size = img.shape[:2]
        img = cv2.resize(img, (1024, 1024))

        # In generated mask, value 1 denotes valid target region, and value 255 stands for region ignored during evaluaiton.
        mask, comments, is_sentence, answer = get_mask_from_json(json_path, img, original_size)

        if mask is not None:
            np.random.seed(0)  # For consistent color selection
            colors = np.random.randint(0, 255, size=(mask.shape[0], 3))
            vis_img = img.copy()

            for i in range(mask.shape[0]):
                # visualization. Green for target, and red for ignore.
                valid_mask = (mask[i] == 1).astype(np.float32)[:, :, None]
                ignore_mask = (mask[i] == 255).astype(np.float32)[:, :, None]

                # Apply colors for each instance based on the mask
                color = colors[i]
                vis_img = vis_img * (1 - valid_mask) * (1 - ignore_mask) + (
                    (color * 0.6 + img * 0.4) * valid_mask
                    + (np.array([255, 0, 0]) * 0.6 + img * 0.4) * ignore_mask
                )

            vis_img = np.concatenate([img, vis_img], 1)
            
            cv2.imwrite(vis_path, vis_img[:, :, ::-1])
            print("Visualization has been saved to: ", vis_path)

        else:
            print("mask is none.")
