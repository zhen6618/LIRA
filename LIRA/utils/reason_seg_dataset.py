import glob
import json
import os
import random
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
                    EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
                    SHORT_QUESTION_LIST)


class ReasonSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
    ):
        self.exclude_val = exclude_val
        self.reason_seg_data = reason_seg_data
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        reason_seg_data, splits = reason_seg_data.split("|")
        splits = splits.split("_")

        # images = []
        # for split in splits:
        #     images_split = glob.glob(
        #         os.path.join(
        #             base_image_dir, "reason_seg", reason_seg_data, split, "*.png"
        #         )
        #     )
        #     images.extend(images_split)
        # jsons = [path.replace(".png", ".json") for path in images]
        # self.reason_seg_data = (images, jsons)

        jsons = []
        for split in splits:
            jsons_split = glob.glob(
                os.path.join(
                    base_image_dir, "reason_seg", reason_seg_data, split, "*.json"
                )
            )
            jsons.extend(jsons_split)

        # images = [path.replace(".json", ".png") for path in jsons]
        # self.reason_seg_data = (images, jsons)

        print("number of reason_seg samples (jsons): ", len(jsons))
        print("self.samples_per_epoch: ", self.samples_per_epoch)

        if explanatory != -1:
            self.explanatory_question_list = EXPLANATORY_QUESTION_LIST  # 指定问题模板列表
            # self.img_to_explanation = {}
            # with open(os.path.join(base_image_dir, "reason_seg", reason_seg_data, "explanatory", "train.json")) as f:
            #     items = json.load(f)

            # for item in items:
            #     img_name = item["image"]
            #     self.img_to_explanation[img_name] = {
            #         "query": item["query"],
            #         "outputs": item["outputs"],
            #     }

            images = []
            for j_i, json_item in enumerate(jsons):
                # print("j_i: ", j_i)
                try:
                    with open(json_item, "r") as json_r:
                        anno_json = json.loads(json_r.read())
                    images.append(anno_json["image"])

                except json.JSONDecodeError as e:
                    # 捕获 JSON 解码错误
                    print(f"JSONDecodeError in file {json_item}: {e}")

            self.reason_seg_data = (images, jsons)

            print("number of reason_seg samples (images): ", len(images))

            # print("len(self.img_to_explanation): ", len(self.img_to_explanation))

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        images, jsons = self.reason_seg_data
        idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        json_path = jsons[idx]

        image = cv2.imread(image_path)
        original_size = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (1024, 1024))

        ori_size = image.shape[:2]
        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        mask, sents, is_sentence, ans = get_mask_from_json(json_path, image, original_size)  # mask: [num, H, W]
        if len(sents) >= self.num_classes_per_sample:  # 最多随机取self.num_classes_per_sample个
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents))) 
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        if mask is None:
            sampled_masks = None
        else:
            sampled_masks = [(mask == 1).astype(np.float32) for _ in range(len(sampled_inds))]

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        # if self.explanatory != -1 and image_path in self.img_to_explanation:
        if self.explanatory != -1:
            if random.random() < self.explanatory:
                choice = 2
            else:
                choice = random.randint(0, 1)

        questions = []
        answers = []
        for text in sampled_sents:
            if is_sentence:
                question_template = random.choice(self.long_question_list)
                questions.append(question_template.format(sent=text))  # '<image>\nIn a game of darts, players often aim for a specific target to accumulate points. What is the target that players try to hit in this picture? Please respond with segmentation mask.'
            else:
                question_template = random.choice(self.short_question_list)
                questions.append(question_template.format(class_name=text.lower()))

            # add explanation if applicable
            # if self.explanatory != -1 and image_path in self.img_to_explanation:
            if self.explanatory != -1:
                choice = 2
                if choice == 0:  # [SEG] token
                    # answers.append(random.choice(self.answer_list))  # 直接往里加<seg>模板 e.g., It is [SEG].
                    answers.append("[SEG] and [SEG].")
                elif choice == 1:  # [SEG] token + text answer
                    # answer = self.img_to_explanation[image_path]["outputs"]
                    answer = ans
                    answer = random.choice(self.answer_list) + " {}".format(answer)
                    questions[-1] = (
                        DEFAULT_IMAGE_TOKEN
                        + "\n"
                        + text
                        + " {}".format(random.choice(self.explanatory_question_list))
                    )
                    answers.append(answer)
                elif choice == 2:  # vanilla text answer
                    # answer = self.img_to_explanation[image_path]["outputs"]
                    answer = ans
                    questions[-1] = DEFAULT_IMAGE_TOKEN + "\n" + text + " Please output segmentation masks."
                    answers.append(answer)

                else:
                    raise ValueError("Not implemented yet.")
            # else:
            #     answers.append(random.choice(self.answer_list))

            conversations = []
            conv = conversation_lib.default_conversation.copy()
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

            i = 0
            while i < len(questions):
                conv.messages = []
                conv.append_message(conv.roles[0], questions[i])
                conv.append_message(conv.roles[1], answers[i])
                conversations.append(conv.get_prompt())
                i += 1

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        if (
            self.explanatory != -1
            # and image_path in self.img_to_explanation
            and mask is None
        ):  
            masks = torch.rand(0, *ori_size)  # (0, 1024, 1024)
            label = torch.ones(ori_size) * self.ignore_label
        else:
            masks = np.concatenate(sampled_masks, axis=0)
            masks = torch.from_numpy(masks)
            label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_sents,
        )
