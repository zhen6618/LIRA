import glob
import json
import os
import time
import cv2
import numpy as np
import re
import uuid
from PIL import Image
from skimage.transform import resize
from copy import deepcopy

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
        if "flag" == label_id.lower():  # meaningless deprecated annotations
            continue

        if points == None:
            masks = None
            continue
        
        masks = []
        for pot in points:  # There are as many instances as there are POTS
            mask = np.zeros((original_size[0], original_size[1]), dtype=np.uint8)
            for polygon in pot:
                polygon = np.array(polygon).reshape(-1, 2)  # Transform the flattened points into the shape of (x, y)
                cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)  # 1 indicates that the foreground is filled with a polygonal area on the mask
            mask = resize(mask.astype('float'), (1024, 1024), order=1, mode="constant", anti_aliasing=False)  # After resize, it becomes a floating-point number
            mask[mask >= 0.5] = 1
            mask[mask < 0.5] = 0
            masks.append(mask)

        # # Sort the masks in the order from top left to bottom right
        # if len(masks) > 1:
        #     # Calculate the center point of the mask
        #     def get_center(mask):
        #         coords = np.argwhere(mask == 1)  # Find the coordinates of all non-zero points in the mask
        #         center = coords.mean(axis=0)  
        #         return tuple(center)

        #     masks = sorted(masks, key=get_center)  # Sort according to the y and x coordinates of the center point, from top left to bottom right
        #     # for m in masks:
        #     #     print(get_center(m))  # Output the coordinates of the center point and check if they are arranged in sequence

        for mi in range(len(masks)):
            masks[mi] = np.expand_dims(masks[mi], axis=0)
        masks = np.concatenate(masks, axis=0)
    # import pdb;pdb.set_trace()

    '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   Open vocabulary   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
    # open_vocab = True
    # if open_vocab:
    #     comments = deepcopy(anno["outputs"])  # Use regular expressions to remove [SEG] and the Spaces before and after it
    #     comments = re.sub(r'\s*\[SEG\]\s*', '', comments)
    #     comments = [comments]

    return masks, comments, is_sentence, answer

furniture_dict = {
    'cabinet': '柜子',
    'bed': '床',
    'chair': '椅子',
    'sofa': '沙发',
    'table': '桌子',
    'bookshelf': '书架',
    'picture': '画',
    'counter': '柜台',
    'desk': '书桌',
    'curtain': '窗帘',
    'refrigerator': '冰箱',
    'shower curtain': '浴帘',
    'toilet': '马桶',
    'sink': '水槽',
    'bathtub': '浴缸'
}

def generate_uuid():
    return str(uuid.uuid4())

# Convert the mask to a polygon
def mask_to_polygon(mask):
    _, binary_mask = cv2.threshold(mask * 255, 127, 255, cv2.THRESH_BINARY)
    
    binary_mask = binary_mask.astype(np.uint8)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        longest_contour = max(contours, key=lambda contour: cv2.arcLength(contour, closed=True))
        
        approx_polygon = cv2.approxPolyDP(longest_contour, epsilon=2, closed=True)
        return approx_polygon

    return []  

# Convert to JSON format
def mask_to_json(mask, mask_type, text, template):
    mask = mask.astype(np.uint8)
    elements = []

    for i in range(mask.shape[0]):
        polygon = mask_to_polygon(mask[i])
        
        polygon_points = [{"x": float(pt[0][0]), "y": float(pt[0][1])} for pt in polygon]

        cur_infos = {
            "attribute": {"type": mask_type[i]},
            "uuid": generate_uuid(),
            "markType": "area",
            "typeId": "area_1",
            "text": text[i],
            "points": polygon_points,
            "rotation": 0,
        }
        elements.append(cur_infos)

    # output = {'result': [{'elements': elements}]}

    return elements


if __name__ == "__main__":
    is_dir = False

    if is_dir:
        # data_dir = "/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/reason_seg/Scannet_2D_Seg_base/val"
        # vis_dir = "/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/reason_seg/Scannet_2D_Seg_base/vis"
        data_dir = "/home/users/zhouzhen05/LISA/dataset/reason_seg/Scannet_2D_Seg_base/val_all"
        vis_dir = "/home/users/zhouzhen05/LISA/dataset/reason_seg/Scannet_2D_Seg_base/vis"
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        # "Template
        with open('2D_answer.json', 'r', encoding='utf-8') as file:
            template = json.load(file)

        json_path_list = sorted(glob.glob(data_dir + "/*.json"))
        import random
        random.seed(42)
        json_path_list = random.sample(json_path_list, 10000)

        count = 0
        # mapping_left = []
        # mapping_right = []

        image_names =[]
        image_links = []
        image_attributes = []

        for json_path in json_path_list:
            with open(json_path, "r") as r:
                anno_json = json.loads(r.read())

            # img_path = json_path.replace(".json", ".png")
            img_path = anno_json['image']
            img_path = img_path.replace("/root/paddlejob/workspace/env_run/zhouzhen05/Data/zhouzhen05/Data", "/home/users/zhouzhen05/Reason_Seg_Data")
            img = cv2.imread(img_path)[:, :, ::-1]
            original_size = img.shape[:2]
            img = cv2.resize(img, (1024, 1024))

            # In generated mask, value 1 denotes valid target region, and value 255 stands for region ignored during evaluaiton.
            mask, comments, is_sentence, answer = get_mask_from_json(json_path, img, original_size)

            if mask is not None:
                count += 1
                if count > 2000:
                    print("Over images")
                    break
                
                '-----------------------------  For annotation start  -----------------------------'
                text_label = [match.strip() for match in re.findall(r'\b(\w+)\s(?=\[SEG\])', anno_json['outputs'])]
                text_label_CN = [furniture_dict[text_l] for text_l in text_label] 

                elements = mask_to_json(mask, mask_type=text_label, text=text_label_CN, template=template)
                json_out = deepcopy(template)
                json_out['result'][0]['elements'] = elements
                json_out['result'][0]['relations']['inclusion'] = []
                json_out['result'][0]['size']['width'] = 1024
                json_out['result'][0]['size']['height'] = 1024

                image_names.append(f'{count}.jpg')
                image_links.append([])
                image_attributes.append(json_out)

                '-----------------------------  For annotation end -----------------------------'

                np.random.seed(0)  # For consistent color selection
                colors = np.random.randint(0, 255, size=(mask.shape[0], 3))
                vis_img = img.copy()

                origin_img_path = os.path.join(vis_dir, f'{count}.jpg')
                image = Image.fromarray(img)
                image.save(origin_img_path)

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
                
                vis_path = os.path.join(vis_dir, f'anno_{count}.jpg')
                cv2.imwrite(vis_path, vis_img[:, :, ::-1])
                print("Visualization has been saved to: ", vis_path)

                # mapping_left.append(f'{count}.jpg') 
                # mapping_right.append(text_label)

        '-----------------------------  For annotation start  -----------------------------'
        with open(os.path.join(vis_dir, 'infos.txt'), "w", encoding="utf-8") as file:
            for i in range(len(image_names)):
                attributes_str = json.dumps(image_attributes[i], ensure_ascii=False)
                
                file.write(f"{image_names[i]}   {image_links[i]}   {attributes_str}\n")
        '-----------------------------  For annotation end -----------------------------'

        # with open(os.path.join(vis_dir, 'mapping.txt'), 'w', encoding='utf-8') as file:
        #     for l, r in zip(mapping_left, mapping_right):
        #         line_content = f"{l}{' ' * 5}{r}\n"
        #         file.write(line_content)

    else:
        base_path = "/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/reason_seg/Scannet_2D_Seg_base_new/val"
        json_path = os.path.join(base_path, "50_root_paddlejob_workspace_env_run_zhouzhen05_Data_zhouzhen05_Data_scans_scene0107_00_color_color_2014.json")
        vis_path = "chat_label_watch.png"

        with open(json_path, "r") as r:
            anno_json = json.loads(r.read())

        instruction = anno_json['text']
        output = anno_json['outputs']
        print("instruction: ", instruction)
        print("output: ", output)

        # img_path = json_path.replace(".json", ".png")
        img_path = anno_json['image']
        print("img_path: ", img_path)
        img = cv2.imread(img_path)[:, :, ::-1]
        original_size = img.shape[:2]
        img = cv2.resize(img, (1024, 1024))

        # In generated mask, value 1 denotes valid target region, and value 255 stands for region ignored during evaluaiton.
        mask, comments, is_sentence, answer = get_mask_from_json(json_path, img, original_size)
        print('comments: ', comments)
        print('answer: ', answer)

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
                    + (np.array([255, 0, 0]) * 0.8 + img * 0.2) * ignore_mask
                )

            vis_img = np.concatenate([img, vis_img], 1)
            
            cv2.imwrite(vis_path, vis_img[:, :, ::-1])
            print("Visualization has been saved to: ", vis_path)

        else:
            print("mask is none.")


    # /root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/reason_seg/Scannet_2D_Seg_base/human_annotaion

    '-----------------------------  ScanNetV2的2D Segmentation Annotation  -----------------------------' 
    # custom_colors = [
    #     (255, 0, 0),        # 红色
    #     (0, 255, 0),        # 绿色
    #     (0, 0, 255),        # 蓝色
    #     (255, 255, 0),      # 黄色
    #     (255, 0, 255),      # 品红色
    #     (0, 255, 255),      # 青色
    #     (255, 128, 0),      # 橙色
    #     (128, 0, 255),      # 紫色
    #     (0, 128, 255),      # 天蓝色
    #     (255, 255, 128),    # 浅黄色
    #     (128, 255, 128),    # 浅绿色
    #     (128, 128, 255),    # 浅蓝色
    #     (255, 128, 128),    # 浅红色
    #     (128, 255, 255),    # 浅青色
    #     (255, 128, 255),    # 浅品红色
    #     (192, 192, 192),    # 灰色
    #     (255, 165, 0),      # 橘黄色
    #     (0, 255, 128),      # 绿色-蓝色
    #     (128, 255, 255)     # 淡青色
    # ]

    # image = cv2.imread(img_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # scene_name = img_path.split('/color/')[0].split('/')[-1]
    # img_id = img_path.split('.jpg')[0].split('_')[-1]

    # scannet_anno_path = f'/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_2d_annotations/{scene_name}/instance-filt/{img_id}.png'
    # label_image = Image.open(scannet_anno_path)
    # label_array = np.array(label_image)

    # unique_label_array = np.unique(label_array)
    # label_mapping = {val: i + 1 for i, val in enumerate(unique_label_array)}  
    # label_array = np.vectorize(label_mapping.get)(label_array)

    # color_label = np.zeros((label_array.shape[0], label_array.shape[1], 3), dtype=np.uint8)
    # for i in range(len(custom_colors)):
    #     color_label[label_array == i] = custom_colors[i]

    # alpha = 0.6
    # overlayed = cv2.addWeighted(image, 1 - alpha, color_label, alpha, 0)

    # Image.fromarray(overlayed).save('chat_label_watch_scannet.png')
    

    # print("Done.")

    '-----------------------------  SAM point segmentation -----------------------------'
    image = Image.open('vs_scannet_3.png')

    sam_image = image.convert("RGB")
    sam_image = np.array(sam_image)

    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry["vit_h"](checkpoint="/root/paddlejob/workspace/env_run/zhouzhen05/Code/DRecon/LLaMA_Factory/generate_color_label/sam_vit_h_4b8939.pth")
    sam.to(device="cuda:1")
    sam_predictor = SamPredictor(sam)

    sam_predictor.set_image(sam_image)

    # sam_input_point = np.array([[500, 375], [1125, 625]])
    sam_input_point = np.array([[300, 300], [600, 100], [450, 200]])  # (x, y)

    # Gradually add points and update the mask
    input_point = np.zeros((0, 2)) 
    input_label = np.zeros((0))

    # Store the initial mask
    sam_masks = None
    sam_masks_input = None

    for sam_i in range(len(sam_input_point)):
        # Add new points
        input_point = np.concatenate((input_point, [sam_input_point[sam_i]]), axis=0)
        input_label = np.concatenate((input_label, [1]), axis=0)  # 1 represents a positive point and 0 represents a negative point

        if sam_masks is None:  # If it is the first time to add a point, then proceed with the reasoning directly
            sam_masks, scores, logits = sam_predictor.predict(
                point_coords=input_point,       
                point_labels=input_label,        
                multimask_output=True            
            )
        elif sam_i != len(sam_input_point) - 1:   # Use the previous mask to reason with the current point
            sam_masks_input = logits[np.argmax(scores), :, :] 
            sam_masks, scores, logits = sam_predictor.predict(
                point_coords=input_point,        
                point_labels=input_label,       
                mask_input=sam_masks_input[None, :, :]
                multimask_output=True           
            )
        else:  # The last reasoning
            sam_masks_input = logits[np.argmax(scores), :, :]  # Choose the best mask from the last time
            sam_masks, scores, logits = sam_predictor.predict(
                point_coords=input_point,        
                point_labels=input_label,        
                mask_input=sam_masks_input[None, :, :]
                multimask_output=False            
            )

    sam_mask = sam_masks[0]

    alpha = 0.5

    image_data = np.array(image)

    red_image = np.zeros_like(image)
    red_image[:, :, 0] = 255
    masked_red_image = red_image * sam_mask[:, :, None]

    overlayed = cv2.addWeighted(image_data, 1 - alpha, masked_red_image, alpha, 0)
    overlayed = cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)

    point_color = (0, 255, 0)  # 绿色
    point_radius = 5  # 点的半径

    # # Draw points on the overlayed image
    # for point in sam_input_point:
    #     x, y = point
    #     cv2.circle(overlayed, (x, y), point_radius, point_color, -1)  

    cv2.imwrite("sam_point.png", overlayed)

    print('Done.')

