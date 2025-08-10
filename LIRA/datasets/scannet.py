import os
import numpy as np
import pickle
import cv2
from PIL import Image
from torch.utils.data import Dataset
from copy import deepcopy

class ScanNetDataset(Dataset):
    def __init__(self, datapath, mode, transforms, nviews, n_scales, grounding_img_idx, selected_img_ids):
        super(ScanNetDataset, self).__init__()
        # self.datapath = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3' 
        self.datapath = '/dev/shm'
        self.instructions_num = 100 

        self.mode = mode
        self.n_views = nviews
        self.transforms = transforms
        self.tsdf_file = 'all_tsdf_{}_1'.format(self.n_views)  # 'all_tsdf_9_1'

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()
        if mode == 'test':
            self.source_path = 'scans_test'
        else:
            self.source_path = 'scans'

        self.n_scales = n_scales
        self.epoch = None
        self.tsdf_cashe = {}
        self.rgb_cashe = {}
        self.semantic_cashe = {}
        self.instance_cashe = {}
        self.max_cashe = 1
        self.grounding_img_idx = grounding_img_idx

        self.selected_img_ids = selected_img_ids

    def build_list(self):
        with open(os.path.join(self.datapath, self.tsdf_file, 'fragments_train.pkl'), 'rb') as f:
            metas_train = pickle.load(f)
        with open(os.path.join(self.datapath, self.tsdf_file, 'fragments_val.pkl'), 'rb') as f:
            metas_val = pickle.load(f)
        metas = metas_train + metas_val

        composite = False
        if not composite:
            with open('datasets/base_scenes_train.txt', 'r') as file:
                selected_scenes = [line.strip() for line in file]
            selected_metas = [item for item in metas if item['scene'] in selected_scenes]

            invalid_scene_instructions = [('scene0000_00', 36, 50)]  # scene_name, fragment_id, instruction_id "Eliminate some strange cases"
            # invalid_scene_instructions = []

            # Copy the instructions for each loop's scans
            final_metas = []
            for i in range(0, self.instructions_num): 
                # add currrent instruction id
                for j in range(len(selected_metas)):
                    meta_tmp = deepcopy(selected_metas[j])
                    meta_tmp['instruction_id'] = i

                    if (meta_tmp['scene'], meta_tmp['fragment_id'], meta_tmp['instruction_id']) not in invalid_scene_instructions:
                        qa_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/grounding_scene_qa_infos_base_new/'
                        if os.path.exists(qa_path + meta_tmp['scene'] + '_' + str(meta_tmp['instruction_id']) + '_qa.pkl'):
                            final_metas.append(meta_tmp)

        else:
            with open('datasets/base_scenes_train.txt', 'r') as file:  
                selected_scenes_base = [line.strip() for line in file]
            selected_metas_base = [item for item in metas if item['scene'] in selected_scenes_base]

            with open('datasets/scannetv2_train.txt', 'r') as file:  
                selected_scenes_extension = [line.strip() for line in file]                
            selected_metas_extension = [item for item in metas if item['scene'] in selected_scenes_extension]


            invalid_scene_instructions = [('scene0000_00', 36, 50)]  # scene_name, fragment_id, instruction_id "Eliminate some strange cases"
            # invalid_scene_instructions = []

            # Copy the instructions for each loop's scans
            final_metas = []

            for i in range(0, 100): 
                # add currrent instruction id
                for j in range(len(selected_metas_base)):
                    meta_tmp = deepcopy(selected_metas_base[j])
                    meta_tmp['instruction_id'] = i

                    if (meta_tmp['scene'], meta_tmp['fragment_id'], meta_tmp['instruction_id']) not in invalid_scene_instructions:
                        qa_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/grounding_scene_qa_infos_base_new/'
                        if os.path.exists(qa_path + meta_tmp['scene'] + '_' + str(meta_tmp['instruction_id']) + '_qa.pkl'):
                            final_metas.append(meta_tmp)

            for i in range(0, 10): 
                # add currrent instruction id
                for j in range(len(selected_metas_extension)):
                    meta_tmp = deepcopy(selected_metas_extension[j])
                    meta_tmp['instruction_id'] = i

                    if (meta_tmp['scene'], meta_tmp['fragment_id'], meta_tmp['instruction_id']) not in invalid_scene_instructions:
                        qa_path = '/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding/grounding_scene_qa_infos_hard/'
                        if os.path.exists(qa_path + meta_tmp['scene'] + '_' + str(meta_tmp['instruction_id']) + '_qa.pkl'):
                            final_metas.append(meta_tmp)                          

        return final_metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filepath, vid):
        intrinsics = np.loadtxt(os.path.join(filepath, 'intrinsic', 'intrinsic_color.txt'), delimiter=' ')[:3, :3]
        intrinsics = intrinsics.astype(np.float32)

        pose_path = "pose_" + "{:d}".format(vid)
        extrinsics = np.loadtxt(os.path.join(filepath, 'pose', pose_path + '.txt'))
        return intrinsics, extrinsics

    def read_img(self, filepath):
        img = Image.open(filepath)
        return img

    def read_depth(self, filepath):
        # Read depth image and camera pose
        depth_im = cv2.imread(filepath, -1).astype(
            np.float32)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > 3.0] = 0
        return depth_im

    def read_scene_volumes(self, data_path, scene):
        if scene not in self.tsdf_cashe.keys():
            if len(self.tsdf_cashe) > self.max_cashe:
                self.tsdf_cashe = {}
            full_tsdf_list = []
            for l in range(self.n_scales + 1):
                # load full tsdf volume
                full_tsdf = np.load(os.path.join(data_path, scene, 'full_tsdf_layer{}.npz'.format(l)),
                                    allow_pickle=True)
                full_tsdf_list.append(full_tsdf.f.arr_0)
            self.tsdf_cashe[scene] = full_tsdf_list
        return self.tsdf_cashe[scene]

    def read_scene_volumes_panoptic(self, data_path, scene):
        if scene not in self.tsdf_cashe.keys():
            if len(self.tsdf_cashe) > self.max_cashe:
                self.tsdf_cashe = {}
                self.rgb_cashe = {}
                self.semantic_cashe = {}
                self.instance_cashe = {}

            full_tsdf_list = []
            full_rgb_list = []
            full_semantic_list = []
            full_insatnce_list = []

            # load full tsdf volume
            full_tsdf = np.load(os.path.join(data_path, scene, 'full_tsdf_layer{}.npz'.format(0)), allow_pickle=True)
            full_tsdf_list.append(full_tsdf.f.arr_0)
            full_rgb = np.load(os.path.join(data_path, scene, 'full_rgb_layer{}.npz'.format(0)), allow_pickle=True)
            full_rgb_list.append(full_rgb.f.arr_0)
            full_semantic = np.load(os.path.join(data_path, scene, 'full_semantic_layer_interpolate{}.npz'.format(0)), allow_pickle=True)
            full_semantic_list.append(full_semantic.f.arr_0)
            full_instance = np.load(os.path.join(data_path, scene, 'full_instance_layer_interpolate{}.npz'.format(0)), allow_pickle=True)
            full_insatnce_list.append(full_instance.f.arr_0)

            self.tsdf_cashe[scene] = full_tsdf_list
            self.rgb_cashe[scene] = full_rgb_list
            self.semantic_cashe[scene] = full_semantic_list
            self.instance_cashe[scene] = full_insatnce_list

        return self.tsdf_cashe[scene], self.rgb_cashe[scene], self.semantic_cashe[scene], self.instance_cashe[scene]

    def __getitem__(self, idx):
        meta = self.metas[idx]

        # # get qa
        # qa_path = self.qa_path + str(meta['scene']) + "_" + str(meta['instruction_id']) + "_qa.pkl"
        # with open(qa_path, 'rb') as file:
        #     scene_qa = pickle.load(file)
        #     qa = scene_qa[0]
        #     qa['instruction'] = qa['instruction'] + ' Please output segmentation masks.'

        # with open(self.mapping_save_path + str(meta['scene']) + '.pkl', 'rb') as file:
        #     ins_mapping_dict = pickle.load(file)

        #     for i in range(len(qa['instance ids'])):
        #         qa['instance ids'][i] = ins_mapping_dict[str(qa['instance ids'][i])]

        #     for j in range(len(qa['candidate'])):
        #         qa['candidate'][j] = ins_mapping_dict[str(qa['candidate'][j])]   

        imgs = []
        depth = []
        extrinsics_list = []
        intrinsics_list = []

        tsdf_list, rgb_list, semantic_list, instance_list = self.read_scene_volumes_panoptic(os.path.join(self.datapath, self.tsdf_file), meta['scene'])

        img_length = len(meta['image_ids'])

        for i, vid in enumerate(meta['image_ids']):
            # load images
            if i in self.selected_img_ids:
                color_path = "color_" + "{:d}".format(vid)
                imgs.append(self.read_img(os.path.join(self.datapath, self.source_path, meta['scene'], 'color', color_path + '.jpg')))

            depth_path = "depth_" + "{:d}".format(vid)
            depth.append(self.read_depth(os.path.join(self.datapath, self.source_path, meta['scene'], 'depth', depth_path + '.png')))

            # load intrinsics and extrinsics
            intrinsics, extrinsics = self.read_cam_file(os.path.join(self.datapath, self.source_path, meta['scene']), vid)

            intrinsics_list.append(intrinsics)
            extrinsics_list.append(extrinsics)

            if i == self.grounding_img_idx:
                grounding_img_path = os.path.join(self.datapath, self.source_path, meta['scene'], 'color', color_path + '.jpg')

        intrinsics = np.stack(intrinsics_list)
        extrinsics = np.stack(extrinsics_list)

        items = {
            'imgs': imgs,
            'depth': depth,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'tsdf_list_full': tsdf_list,
            'rgb_list_full': rgb_list,
            'semantic_list_full': semantic_list,
            'instance_list_full': instance_list,
            'vol_origin': meta['vol_origin'],
            'old_origin': meta['vol_origin'],
            'scene': meta['scene'],
            'fragment': meta['scene'] + '_' + str(meta['fragment_id']),
            'epoch': [self.epoch],
            'grounding_img_path': grounding_img_path,
            # 'qa': qa,  # DDP collate_fn cannot be spliced
            'instruction_id': meta['instruction_id'],
        }

        if self.transforms is not None:
            # print('data augmentation', '*' * 50)
            items = self.transforms(items)
        return items
