import os
import pickle
import time

with open('generate_color_label/scene_train_val.txt', 'r') as file:
    lines = file.readlines()
scene_names = [line.strip() for line in lines]

data_root_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data/zhouzhen05/Data/scans'
qa_path = "/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/grounding_scene_qa_infos_hard/"
gt_ins_infos_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/grounding_scene_instance_infos/'

init_pkl_files = [f for f in os.listdir(qa_path) if f.endswith('.pkl')]
pkl_files = []
instruction_num = 10
for pkl_file in init_pkl_files:
    cur_instruction = pkl_file.split('_qa')[0].split('_')[-1]
    if int(cur_instruction) < instruction_num:
        pkl_files.append(pkl_file)
print('total qa files: ', len(pkl_files))
time.sleep(5)

classes = [
    'cabinet', 'bed', 'chair', 'sofa', 'table', 'bookshelf', 
    'picture', 'counter', 'desk', 'curtain', 'refrigerator', 
    'shower curtain', 'toilet', 'sink', 'bathtub'
]

spatial_realtions = [
    "closest", "nearest", 
    "farthest", "most distant", "most remote", 
    "second closest", "next closest", "second nearest", 
    "second farthest", "next farthest", "second remotest", 
    "largest", "most sizable", "biggest", 
    "smallest", "tiniest", 
    "second largest", "next to largest", "second biggest", 
    "second smallest", "next to smallest", "second tiniest"
]

# ------------------------   Statistics annotated images   ------------------------
annotated_images = []
count = 0
for scene_name in scene_names:
    count += 1
    if count % 100 == 0:
        print("Processed {} scenes".format(count))
    image_dir_path = os.path.join(data_root_path, scene_name, 'color')
    images = os.listdir(image_dir_path)
    annotated_images = annotated_images + images

implict_instructions = []
total_targets = []
multi_class = []
multi_targets = []
zero_targets = []
spatial_realtion_instructions = []
length_instructions = []
all_instructions = []

count = 0
k = []

for pkl_file in pkl_files:
    count += 1
    if count % 100 == 0:
        print("Processed {} files".format(count))

    scene_name = pkl_file[:12]

    file_path = os.path.join(qa_path, pkl_file)
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        data = data[0]

    instruction = data['instruction']
    gt_ids = data['instance ids']
    k.append(len(gt_ids))

    with open(gt_ins_infos_path + scene_name + '_instance.pkl', 'rb') as file: 
        gt_ins_infos = pickle.load(file)
    gt_cur_infos = [gt_ins_infos[i] for i in gt_ids]

    # ------------------------   Implicit instructions   ------------------------
    is_implict_instruction = any(cls.lower() in instruction.lower() for cls in classes)
    if is_implict_instruction:
        implict_instructions.append(instruction)

    # ------------------------   targets   ------------------------
    total_targets = total_targets + gt_ids

    # ------------------------   Multi-class   ------------------------
    cur_cls = []
    for ins_info in gt_cur_infos:
        cur_cls.append(ins_info['semantic label'])
    num_cur_cls = len(set(cur_cls))
    multi_class.append(num_cur_cls > 1)

    # ------------------------   Multi-target   ------------------------
    multi_targets.append(len(gt_ids) > 1)

    # ------------------------   Zero-target   ------------------------
    zero_targets.append(len(gt_ids) == 0)

    # ------------------------   Spatial relations   ------------------------
    is_spatial_relation = any(relation.lower() in instruction.lower() for relation in spatial_realtions)
    if is_spatial_relation:
        spatial_realtion_instructions.append(instruction)

    # ------------------------   Length instructions   ------------------------
    length_instructions.append(len(instruction.split()))

    # ------------------------   instruction examples   ------------------------
    all_instructions.append(instruction)


print('')
print('')

print('implict instruction num: ', len(implict_instructions))
print('total targets: ', len(total_targets))
print('annotated images: ', len(annotated_images))
print('multi-class: ', sum(multi_class))
print('multi-target: ', sum(multi_targets))
print('zero-target: ', sum(zero_targets))
print('spatial relation: ', len(spatial_realtion_instructions))
print('length instructions: ', sum(length_instructions) / len(length_instructions))

# # instruction examples
# for j, ins in enumerate(all_instructions[1000:1200]):
#     print(f"{j}. ", ins)

print('')
print('')
