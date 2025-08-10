import argparse
import os
import sys

import cv2
import numpy as np
from copy import deepcopy
from scipy.ndimage import label
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
import matplotlib.pyplot as plt

from models.LISA import LISAForCausalLM
from models.llava import conversation as conversation_lib
from models.llava.mm_utils import tokenizer_image_token
from models.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

def save_mask_heatmap(mask, save_path):
    """
    The result of sigmoid processing of the two-dimensional mask is saved as a heat map.
    """

    plt.figure(figsize=(8, 6))  
    plt.imshow(mask, cmap='hot', interpolation='nearest')
    plt.colorbar() 
    plt.axis('off')  
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0) 
    plt.close() 

    print("{} has been saved.".format(save_path))

class argparser:
    def __init__(self, server_name=None, port=None, model_path=None, path_vision_cfg=None, path_inter_cfg=None):
        # self.version = "/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/LISA_checkpoint/LISA-7B-v1-explanatory_base/"
        # self.version = "/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/LISA_checkpoint/LISA-7B-v1-explanatory_1/"

        self.version = "/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/LISA_checkpoint/LISA-7B-v1-explanatory_base_new/"
        # self.version = "/root/paddlejob/workspace/env_run/zhouzhen05/Code/LISA/checkpoint/LISA-7B-v1-explanatory_base_new_hard_composite"
        # self.version = "/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/LISA_checkpoint/LISA-7B-v1-explanatory_hard/"

        self.vis_save_path = "./vis_output"  
        self.precision = "bf16"

        self.image_size = 1024
        self.model_max_length = 512
        self.lora_r = 8

        self.vision_tower = "openai/clip-vit-large-patch14"
        # self.local_rank = int(os.environ["RANK"])
        self.load_in_8bit = False       

        self.load_in_4bit = False
        self.use_mm_start_end = True
        self.conv_type = "llava_v1"       


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

class Reason_Seg:
    def __init__(self):
        print("Initialize ReasonSeg...")
        args = argparser() 
        self.args = args

        os.makedirs(args.vis_save_path, exist_ok=True)

        # Create model
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.version,
            cache_dir=None,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        args.seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]


        torch_dtype = torch.float32
        if args.precision == "bf16":
            torch_dtype = torch.bfloat16
        elif args.precision == "fp16":
            torch_dtype = torch.half

        kwargs = {"torch_dtype": torch_dtype}
        if args.load_in_4bit:
            kwargs.update(
                {
                    "torch_dtype": torch.half,
                    "load_in_4bit": True,
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        llm_int8_skip_modules=["visual_model"],
                    ),
                }
            )
        elif args.load_in_8bit:
            kwargs.update(
                {
                    "torch_dtype": torch.half,
                    "quantization_config": BitsAndBytesConfig(
                        llm_int8_skip_modules=["visual_model"],
                        load_in_8bit=True,
                    ),
                }
            )

        print("Load from ", args.version, "*" * 100)
        self.model = LISAForCausalLM.from_pretrained(
            args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
        )

        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.get_model().initialize_vision_modules(self.model.get_model().config)
        vision_tower = self.model.get_model().get_vision_tower()
        vision_tower.to(dtype=torch_dtype)

        if args.precision == "bf16":
            self.model = self.model.bfloat16().cuda()
        elif (
            args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
        ):
            vision_tower = self.model.get_model().get_vision_tower()
            self.model.model.vision_tower = None
            import deepspeed

            self.model_engine = deepspeed.init_inference(
                model=self.model,
                dtype=torch.half,
                replace_with_kernel_inject=True,
                replace_method="auto",
            )
            self.model = self.model_engine.module
            self.model.model.vision_tower = vision_tower.half().cuda()
        elif args.precision == "fp32":
            self.model = self.model.float().cuda()

        vision_tower = self.model.get_model().get_vision_tower()
        # vision_tower.to(device=self.model.device)

        self.clip_image_processor = CLIPImageProcessor.from_pretrained(self.model.config.vision_tower)
        self.transform = ResizeLongestSide(args.image_size)

        self.model.eval()    

    def inference(self, prompt=None, image_np=None):
        device = self.model.device

        conv = deepcopy(conversation_lib.conv_templates[self.args.conv_type])
        conv.messages = []

        "如果不传入text"
        if prompt is None:
            prompt = input("Please input your prompt: ")

        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        if self.args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        "如果不传入图片"
        if image_np is None:
            image_path = input("Please input the image path: ")  
            if not os.path.exists(image_path):
                print("File not found in {}".format(image_path))

            image_np = cv2.imread(image_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            image_np = cv2.resize(image_np, (1024, 1024))

        original_size_list = [image_np.shape[:2]]

        image_clip = (
            self.clip_image_processor.preprocess(image_np, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .to(device)
        )
        if self.args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif self.args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        image = self.transform.apply_image(image_np)
        resize_list = [image.shape[:2]]

        image = (
            preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .to(device)
        )
        if self.args.precision == "bf16":
            image = image.bfloat16()
        elif self.args.precision == "fp16":
            image = image.half()
        else:
            image = image.float()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).to(device)

        output_ids, pred_masks, out_text_feats, out_img_feats = self.model.evaluate(
            image_clip,
            image,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=512,
            tokenizer=self.tokenizer,
        )
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

        text_output = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")

        # return
        response = text_output.split('ASSISTANT: ')[1].split("</s>")[0]
        masks = pred_masks[0] > 0

        "*************   Save prediction results   *************"
        # print("text_output: ", text_output)
        # for i, pred_mask in enumerate(pred_masks):
        #     if pred_mask.shape[0] == 0:
        #         continue

        #     num_colors = 10
        #     colors = [np.random.randint(0, 256, size=3) for _ in range(num_colors)]
        #     colors = [np.array(color) for color in colors]
        #     save_img = image_np.copy()
            
        #     for mask_i in range(len(pred_mask)):
        #         pred_m = pred_mask[mask_i].detach().cpu().numpy()
        #         pred_m = pred_m > 0

        #         # save_path = "{}/{}_mask_{}.jpg".format(self.args.vis_save_path, image_path.split("/")[-1].split(".")[0], mask_i)
        #         save_path = os.path.join(self.args.vis_save_path, 'reason_seg_mask.jpg')

        #         cv2.imwrite(save_path, pred_m * 100)
        #         print("{} has been saved.".format(save_path))

        #         # save_path = "{}/{}_masked_img_{}.jpg".format(self.args.vis_save_path, image_path.split("/")[-1].split(".")[0], mask_i)
        #         save_path = os.path.join(self.args.vis_save_path, 'reason_seg_masked_img.jpg')

        #         save_img[pred_m] = (
        #             image_np * 0.5
        #             + pred_m[:, :, None].astype(np.uint8) * colors[mask_i] * 0.5
        #         )[pred_m]

        #     save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite(save_path, save_img)
        #     print("{} has been saved.".format(save_path))

        return response, masks, out_text_feats, out_img_feats
    

    def inference_parallel(self, prompt=None, image_np=None):
        device = self.model.device

        conv = deepcopy(conversation_lib.conv_templates[self.args.conv_type])
        conv.messages = []

        # prompt
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        if self.args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        # image
        original_size_list = [image_np[0].shape[:2]]  # [(1024, 1024)]

        image_clip = []
        for i in range(len(image_np)):
            image_clip.append(self.clip_image_processor.preprocess(image_np[i], return_tensors="pt")["pixel_values"][0].unsqueeze(0).to(device))  # [1, 3, 224, 224]
        image_clip = torch.cat(image_clip, dim=0)  # [N ,3, 224, 224]

        if self.args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif self.args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        image = []
        for i in range(len(image_np)):
            image.append(self.transform.apply_image(image_np[i]))
            resize_list = [image[i].shape[:2]]  # [(1024, 1024)]
            image[i] = (preprocess(torch.from_numpy(image[i]).permute(2, 0, 1).contiguous()).unsqueeze(0).to(device))  # [1, 3, 1024, 1024]
        image = torch.cat(image, dim=0)  # [N ,3, 1024, 1024]
        
        if self.args.precision == "bf16":
            image = image.bfloat16()
        elif self.args.precision == "fp16":
            image = image.half()
        else:
            image = image.float()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).repeat(len(image), 1).to(device)  # [N, 69]

        output_ids, pred_masks, out_text_feats, out_img_feats = self.model.evaluate_parallel(
            image_clip,
            image,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=512,
            tokenizer=self.tokenizer,
        )

        response = []
        for i in range(len(output_ids)):  # [N, 75]
            output_ids_tmp = output_ids[i][output_ids[i] != IMAGE_TOKEN_INDEX]

            text_output = self.tokenizer.decode(output_ids_tmp, skip_special_tokens=False)
            text_output = text_output.replace("\n", "").replace("  ", " ")
            response.append(text_output.split('ASSISTANT: ')[1].split("</s>")[0])

        masks = []
        for i in range(len(pred_masks)):
            masks.append(pred_masks[i] > 0)

        "*************   Save prediction results   *************"
        # for i, response_i in enumerate(response):
        #     print(i, response_i)

        # for i, pred_mask in enumerate(pred_masks):
        #     if pred_mask.shape[0] == 0:
        #         continue

        #     num_colors = 10
        #     colors = [np.random.randint(0, 256, size=3) for _ in range(num_colors)]
        #     colors = [np.array(color) for color in colors]

        #     save_img = image_np[i].copy()
            
        #     for mask_i in range(len(pred_mask)):
        #         pred_m = pred_mask[mask_i].detach().cpu().numpy()
        #         pred_m = pred_m > 0

        #         save_path = os.path.join(self.args.vis_save_path, f'reason_seg_mask_{i}.jpg')

        #         cv2.imwrite(save_path, pred_m * 100)
        #         print("{} has been saved.".format(save_path))

        #         save_path = os.path.join(self.args.vis_save_path, f'reason_seg_masked_img_{i}.jpg')

        #         save_img[pred_m] = (
        #             image_np[i] * 0.5
        #             + pred_m[:, :, None].astype(np.uint8) * colors[mask_i] * 0.5
        #         )[pred_m]

        #     save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite(save_path, save_img)
        #     print("{} has been saved.".format(save_path))

        return response, masks, out_text_feats, out_img_feats
    
    def inference_batch_parallel(self, batch_indexs, prompt_all, image_np):
        device = self.model.device
        num_imgs_per_batch = int(len(image_np) / len(batch_indexs))
        num_FBV = len(batch_indexs)

        # *****  image  *****
        original_size_list = [image_np[0].shape[:2]]  # [(1024, 1024)]

        image_clip = []
        for i in range(len(image_np)):
            image_clip.append(self.clip_image_processor.preprocess(image_np[i], return_tensors="pt")["pixel_values"][0].unsqueeze(0).to(device))  # [1, 3, 224, 224]
        image_clip = torch.cat(image_clip, dim=0)  # [N ,3, 224, 224]

        if self.args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif self.args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        image = []
        for i in range(len(image_np)):
            image.append(self.transform.apply_image(image_np[i]))
            resize_list = [image[i].shape[:2]]  # [(1024, 1024)]
            image[i] = (preprocess(torch.from_numpy(image[i]).permute(2, 0, 1).contiguous()).unsqueeze(0).to(device))  # [1, 3, 1024, 1024]
        image = torch.cat(image, dim=0)  # [N ,3, 1024, 1024]
        
        if self.args.precision == "bf16":
            image = image.bfloat16()
        elif self.args.precision == "fp16":
            image = image.half()
        else:
            image = image.float()

        # *****  prompt  *****
        input_ids = []
        for prompt in prompt_all:
            input_ids.append([])

            conv = deepcopy(conversation_lib.conv_templates[self.args.conv_type])
            conv.messages = []

            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            if self.args.use_mm_start_end:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                )
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], "")
            prompt = conv.get_prompt()

            input_ids[-1] = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
            input_ids[-1] = input_ids[-1].unsqueeze(0).repeat(num_imgs_per_batch, 1).to(device)  # [N, 69]

        "Parallel reasoning"
        # Get the index of a single element
        input_ids_shape = [input_ids_i.shape[1] for input_ids_i in input_ids]
        input_ids_shape_np = np.array(input_ids_shape)
        _, unique_indices = np.unique(input_ids_shape_np, return_index=True)
        unique_elements = input_ids_shape_np[np.sort(unique_indices)]

        # if len(unique_elements) > 1:
        #     print("Different scans")

        final_response, final_masks, final_out_text_feats, final_out_img_feats = [[]] * num_FBV, [[]] * num_FBV, [[]] * num_FBV, [[]] * num_FBV
        for uni_i in unique_elements:
            uni_input_ids = []
            uni_image_clip = []
            uni_image = []

            for idx, input_ids_shape_i in enumerate(input_ids_shape):
                if input_ids_shape_i == uni_i:
                    uni_input_ids.append(input_ids[idx])
                    uni_image_clip.append(image_clip[idx*num_imgs_per_batch:(idx + 1)*num_imgs_per_batch])
                    uni_image.append(image[idx*num_imgs_per_batch:(idx + 1)*num_imgs_per_batch])

            uni_input_ids = torch.cat(uni_input_ids, dim=0)
            uni_image_clip = torch.cat(uni_image_clip, dim=0)
            uni_image = torch.cat(uni_image, dim=0)

            # *****  inference  *****
            output_ids, pred_masks, out_text_feats, out_img_feats = self.model.evaluate_parallel(
                uni_image_clip,
                uni_image,
                uni_input_ids,
                resize_list,
                original_size_list,
                max_new_tokens=512,
                tokenizer=self.tokenizer,
            )       

            response = []
            for i in range(len(output_ids)):  # [N, 75]
                output_ids_tmp = output_ids[i][output_ids[i] != IMAGE_TOKEN_INDEX]

                text_output = self.tokenizer.decode(output_ids_tmp, skip_special_tokens=False)
                text_output = text_output.replace("\n", "").replace("  ", " ")
                response.append(text_output.split('ASSISTANT: ')[1].split("</s>")[0])

            masks = []
            for i in range(len(pred_masks)):
                # save_mask_heatmap(pred_masks[i].sigmoid().squeeze(0).detach().cpu().numpy(), os.path.join(self.args.vis_save_path, f'reason_heatmap_mask_{i}.jpg'))
                masks.append(pred_masks[i].sigmoid() > 0.5)
                # print((pred_masks[i].sigmoid() > 0.8).sum() / (pred_masks[i].sigmoid() > 0.5).sum())

            # Fill in
            count = -1
            for idx, input_ids_shape_i in enumerate(input_ids_shape):
                if input_ids_shape_i == uni_i:
                    count += 1

                    final_response[idx] = response[count*num_imgs_per_batch:(count + 1)*num_imgs_per_batch]
                    final_masks[idx] = masks[count*num_imgs_per_batch:(count + 1)*num_imgs_per_batch]
                    final_out_text_feats[idx] = out_text_feats[count*num_imgs_per_batch:(count + 1)*num_imgs_per_batch]
                    final_out_img_feats[idx] = out_img_feats[count*num_imgs_per_batch:(count + 1)*num_imgs_per_batch]

        # tidy
        final_response = [item for sublist in final_response for item in sublist]
        final_masks = [item for sublist in final_masks for item in sublist]
        final_out_text_feats = [item for sublist in final_out_text_feats for item in sublist]
        final_out_img_feats = torch.cat(final_out_img_feats, dim=0)

        # # Take the largest connected area
        # for mask_i in range(len(final_masks)):
        #     for mask_j in range(len(final_masks[mask_i])):
        #         mask_tmp = keep_two_largest_connected_regions(final_masks[mask_i][mask_j].cpu().numpy())
        #         final_masks[mask_i][mask_j] = torch.from_numpy(mask_tmp).to(device)

        "*************   save   *************"
        # for i, response_i in enumerate(final_response):
        #     print(i, response_i)

        # for i, pred_mask in enumerate(final_masks):
        #     if pred_mask.shape[0] == 0:
        #         save_img = image_np[i].copy()
        #         save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)

        #         save_path = os.path.join(self.args.vis_save_path, f'reason_seg_masked_img_{i}.jpg')
        #         cv2.imwrite(save_path, save_img)
        #         print("{} has been saved.".format(save_path))

        #     else:
        #         # num_colors = 10
        #         # colors = [np.random.randint(0, 256, size=3) for _ in range(num_colors)]
        #         # colors = [np.array(color) for color in colors]
        #         colors = [
        #             np.array([255, 0, 0]),      # 红色
        #             np.array([0, 255, 0]),      # 绿色
        #             np.array([0, 0, 255]),      # 蓝色
        #             np.array([255, 255, 0]),    # 黄色
        #             np.array([255, 0, 255]),    # 紫色
        #             np.array([0, 255, 255]),    # 青色
        #             np.array([128, 0, 0]),      # 深红
        #             np.array([0, 128, 0]),      # 深绿
        #             np.array([0, 0, 128]),      # 深蓝
        #             np.array([255, 128, 0]),    # 橙色
        #             np.array([128, 255, 0]),    # 浅绿色
        #             np.array([0, 255, 128]),    # 浅青色
        #             np.array([255, 0, 128]),    # 粉色
        #             np.array([128, 128, 255]),  # 浅蓝
        #             np.array([128, 255, 255]),  # 浅紫
        #             np.array([255, 255, 128]),  # 浅黄
        #             np.array([128, 0, 255]),    # 深紫
        #             np.array([0, 128, 255])     # 天蓝
        #         ]

        #         save_img = image_np[i].copy()
                
        #         for mask_i in range(len(pred_mask)):
        #             pred_m = pred_mask[mask_i].detach().cpu().numpy()
        #             pred_m = pred_m > 0

        #             save_path = os.path.join(self.args.vis_save_path, f'reason_seg_mask_{i}.jpg')

        #             cv2.imwrite(save_path, pred_m * 100)
        #             print("{} has been saved.".format(save_path))

        #             save_path = os.path.join(self.args.vis_save_path, f'reason_seg_masked_img_{i}.jpg')

        #             save_img[pred_m] = (
        #                 image_np[i] * 0.7
        #                 + pred_m[:, :, None].astype(np.uint8) * colors[mask_i] * 0.3
        #             )[pred_m]

        #         save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
        #         cv2.imwrite(save_path, save_img)
        #         print("{} has been saved.".format(save_path))

        return final_response, final_masks, final_out_text_feats, final_out_img_feats
    

def keep_two_largest_connected_regions(mask):
    labeled_mask, num_features = label(mask)

    if num_features == 0:
        return mask
    
    region_sizes = [(i, np.sum(labeled_mask == i)) for i in range(1, num_features + 1)]
    
    largest_two_regions = sorted(region_sizes, key=lambda x: x[1], reverse=True)[:1]
    
    largest_two_labels = {region[0] for region in largest_two_regions}

    largest_two_mask = np.isin(labeled_mask, list(largest_two_labels))

    return largest_two_mask


def main2():
    reason_seg_model = Reason_Seg()
    reason_seg_model.inference()


if __name__ == "__main__":
    # main(sys.argv[1:])
    main2()
