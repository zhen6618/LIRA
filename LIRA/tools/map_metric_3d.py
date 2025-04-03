# -*- encoding: utf-8 -*-
"""
@File    :   map_metric_3d.py
@Time    :   2024/08/01 13:09:04
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import os
import numpy as np
import torch
import tqdm
import json


class Metric:
    """
    Metric
    """

    def __init__(self, **kwargs):
        pass

    def name(self):
        """
        Return name of metric instance.
        """
        return self.__class__.__name__

    def reset(self):
        """reset"""
        raise NotImplementedError

    def update(self, results):
        """
        update

        Args:
            result (dict|list[dict]): result dict

        Return:
            None
        """
        raise NotImplementedError

    def accumulate(self, save_dir=None) -> dict:
        """
        accumulate

        Args:
            save_dir (str): save dir for metric curve

        Return:
            metric_dict (dict): metric dict
        """
        raise NotImplementedError


class ComposeMetric(Metric):
    """
    Compose Metric
    """

    def __init__(
        self,
        descriptions,
        metrics,
        **kwargs,
    ):
        assert len(descriptions) == len(metrics), "length of descriptions and metrics must be equal"
        self.descriptions = descriptions
        self.metrics = [build_metric(metric) for metric in metrics]

    def reset(self):
        """reset"""
        for metric in self.metrics:
            metric.reset()

    def update(self, results):
        """
        update

        Args:
            result (dict|list[dict]): result dict

        Return:
            None
        """
        for metric in self.metrics:
            metric.update(results)

    def accumulate(self, save_dir=None) -> dict:
        """
        accumulate

        Args:
            save_dir (str): save dir for metric curve

        Return:
            metric_dict (dict): metric dict
        """
        metric_dict = {}
        for desc, metric in zip(self.descriptions, self.metrics):
            tmp_dict = metric.accumulate(save_dir)
            for key, value in tmp_dict.items():
                metric_dict[key + desc] = value

        return metric_dict


def iou_calc(mask_pred, mask_gt):
    """
    计算两个二值 mask 的 IoU
    """
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    if union == 0:
        return 0.0
    return intersection / union


class MAPMetric3D(Metric):
    """
    MAP Metric
    """

    def __init__(
        self,
        class_names,
        iou_thresh=0.5,
        score_thresh=None,
        pc_range=None,
        **kwargs,
    ):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.pc_range = pc_range
        
        self.reset()

    def reset(self):
        """reset"""
        self.tpfp_buffer = [[] for _ in range(self.num_classes)]
        self.pred_score_buffer = [[] for _ in range(self.num_classes)]
        self.gt_count_buffer = [0] * self.num_classes
        self.confusion_matrix = np.zeros((self.num_classes + 1, self.num_classes + 1), dtype="int32")

    def update(self, results):
        """
        update

        Args:
            result (dict|list[dict]): result dict

        Return:
            None
        """
        device = torch.cuda.current_device()
        if self.pc_range is not None:
            self.pc_range = torch.tensor(self.pc_range).to(device)

        if not isinstance(results, list):
            results = [results]

        for result in tqdm.tqdm(results):
            pred_boxes = result["pred_boxes"].to(device)
            pred_scores = result["pred_scores"].to(device)
            pred_labels = result["pred_labels"].to(device).long()
            gt_boxes = result["gt_boxes"].to(device)
            gt_labels = result["gt_labels"].to(device).long()

            # filter by score
            if self.score_thresh is not None and len(pred_boxes):
                mask = pred_scores >= self.score_thresh
                pred_boxes = pred_boxes[mask]
                pred_scores = pred_scores[mask]
                pred_labels = pred_labels[mask]

            # filter by pc_range
            if self.pc_range is not None:
                if len(pred_boxes):
                    mask = ((pred_boxes[:, :3] >= self.pc_range[:3]) & (pred_boxes[:, :3] <= self.pc_range[3:])).all(
                        dim=-1
                    )
                    pred_boxes = pred_boxes[mask]
                    pred_scores = pred_scores[mask]
                    pred_labels = pred_labels[mask]

                if len(gt_boxes):
                    mask = ((gt_boxes[:, :3] >= self.pc_range[:3]) & (gt_boxes[:, :3] <= self.pc_range[3:])).all(dim=-1)
                    gt_boxes = gt_boxes[mask]
                    gt_labels = gt_labels[mask]

            # confusion matrix
            confusion_matrix = torch.zeros(
                (self.num_classes + 1, self.num_classes + 1),
                dtype=torch.int32,
                device=device,
            )
            # GT
            confusion_matrix.index_put_(
                (gt_labels, torch.full_like(gt_labels, self.num_classes)),
                torch.ones_like(gt_labels, dtype=torch.int32),
                accumulate=True,
            )
            # Prediction
            confusion_matrix.index_put_(
                (torch.full_like(pred_labels, self.num_classes), pred_labels),
                torch.ones_like(pred_labels, dtype=torch.int32),
                accumulate=True,
            )
            if len(gt_boxes) and len(pred_boxes):
                ious = iou_calc(pred_boxes, gt_boxes)
                gt_indices = torch.argmax(ious, dim=-1)
                pred_indices = torch.arange(len(pred_boxes), device=device)
                mask1 = ious[pred_indices, gt_indices] >= self.iou_thresh
                mask = torch.zeros_like(ious, dtype=torch.int32)
                mask[pred_indices, gt_indices] = mask1.int()
                cumsum = torch.cumsum(mask, dim=0)
                mask2 = cumsum[pred_indices, gt_indices] == 1
                matched = (mask1 & mask2).bool()
                matched_gt_indices = gt_indices[matched]
                matched_pred_indices = pred_indices[matched]
                matched_gt_labels = gt_labels[matched_gt_indices]
                matched_pred_labels = pred_labels[matched_pred_indices]
                confusion_matrix.index_put_(
                    (matched_gt_labels, matched_pred_labels),
                    torch.ones_like(matched_gt_labels, dtype=torch.int32),
                    accumulate=True,
                )

            confusion_matrix[:, -1] -= confusion_matrix[:, :-1].sum(dim=1)
            confusion_matrix[-1] -= confusion_matrix[:-1].sum(dim=0)
            confusion_matrix[-1, -1] = 0
            self.confusion_matrix += confusion_matrix.cpu().numpy()

            # for each class
            for class_id in range(self.num_classes):
                # record gt count
                mask = gt_labels == class_id
                num_gt = mask.sum()
                if num_gt:
                    cur_gt_boxes = gt_boxes[mask]
                self.gt_count_buffer[class_id] += num_gt.item()

                # record pred score
                mask = pred_labels == class_id
                num_pred = mask.sum()
                if num_pred == 0:
                    continue
                cur_pred_boxes = pred_boxes[mask]
                cur_pred_scores = pred_scores[mask]
                self.pred_score_buffer[class_id].append(cur_pred_scores.cpu().numpy())

                # calculate iou
                tpfp = cur_pred_boxes.new_zeros((num_pred), dtype=torch.int32)
                if num_gt:
                    ious = iou_calc(cur_pred_boxes, cur_gt_boxes)

                    gt_indices = torch.argmax(ious, dim=-1)
                    pred_indices = torch.arange(num_pred, device=device)
                    mask1 = ious[pred_indices, gt_indices] >= self.iou_thresh
                    mask = torch.zeros_like(ious, dtype=torch.int32)
                    mask[pred_indices, gt_indices] = mask1.int()
                    cumsum = torch.cumsum(mask, dim=0)
                    mask2 = cumsum[pred_indices, gt_indices] == 1
                    tpfp = mask1 & mask2
                    tpfp = tpfp.int()

                self.tpfp_buffer[class_id].append(tpfp.cpu().numpy())

    def accumulate(self, save_dir=None) -> dict:
        """
        accumulate

        Args:
            save_dir (str): save dir for metric curve

        Return:
            metric_dict (dict): metric dict
        """
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        # collect results from all ranks
        buffer = dict(
            gt_count_buffer=self.gt_count_buffer,
            tpfp_buffer=self.tpfp_buffer,
            pred_score_buffer=self.pred_score_buffer,
            confusion_matrix=self.confusion_matrix,
        )
        ret_list = [buffer]
        self.reset()
        for ret in ret_list:
            self.confusion_matrix += ret["confusion_matrix"]
            for class_id in range(self.num_classes):
                self.gt_count_buffer[class_id] += ret["gt_count_buffer"][class_id]
                self.tpfp_buffer[class_id].extend(ret["tpfp_buffer"][class_id])
                self.pred_score_buffer[class_id].extend(ret["pred_score_buffer"][class_id])

        # print confusion matrix
        confusion_matrix = self.confusion_matrix.tolist()
        class_names_with_bg = self.class_names + ["BG"]
        keys = ["GT\\Pred"] + class_names_with_bg
        print("| " + " | ".join(keys) + " |")
        print("|:---:" * len(keys) + "|")
        for i, values in enumerate(confusion_matrix):
            values = [class_names_with_bg[i]] + ["{}".format(value) for value in values]
            print("| " + " | ".join(values) + " |")

        ap_dict = dict()
        precision_dict = dict()
        recall_dict = dict()
        save_data = dict()
        for class_id in range(self.num_classes):
            class_name = self.class_names[class_id]
            gt_count = self.gt_count_buffer[class_id]
            if gt_count == 0 or len(self.tpfp_buffer[class_id]) == 0:
                ap_dict[class_name] = 0
                precision_dict[class_name] = 0
                recall_dict[class_name] = 0
                continue

            tpfp = np.concatenate(self.tpfp_buffer[class_id])
            pred_score = np.concatenate(self.pred_score_buffer[class_id])

            # sort
            indices = np.argsort(-pred_score, axis=0)
            tpfp = tpfp[indices]
            pred_score = pred_score[indices]

            tp = np.cumsum(tpfp, axis=0)
            fp = np.cumsum(1 - tpfp, axis=0)
            prs = tp / np.maximum(tp + fp, 1e-9)
            rcs = tp / gt_count
            AP, P, R = self._calc_AP(prs, rcs)
            ap_dict[class_name] = AP
            precision_dict[class_name] = P
            recall_dict[class_name] = R

            save_data[class_name] = dict(
                AP=float(AP),
                P=float(P),
                R=float(R),
                prs=prs.tolist(),
                rcs=rcs.tolist(),
                scores=pred_score.tolist(),
            )

        if save_dir is not None:
            with open(os.path.join(save_dir, "metric.json"), "w") as f:
                json.dump(save_data, f)

        values = np.array(list(ap_dict.values()))
        ap_dict["mAP"] = values.sum() / self.num_classes
        keys = ["mAP"] + self.class_names
        values = [ap_dict[key] for key in keys]
        values = ["{:.2f}".format(value * 100) for value in values]
        print("| " + " | ".join(keys) + " |")
        print("|" + " :---: |" * len(keys))
        print("| " + " | ".join(values) + " |")

        values = np.array(list(precision_dict.values()))
        precision_dict["mP"] = values.sum() / self.num_classes
        keys = ["mP"] + self.class_names
        values = [precision_dict[key] for key in keys]
        values = ["{:.2f}".format(value * 100) for value in values]
        print("| " + " | ".join(keys) + " |")
        print("|" + " :---: |" * len(keys))
        print("| " + " | ".join(values) + " |")

        values = np.array(list(recall_dict.values()))
        recall_dict["mR"] = values.sum() / self.num_classes
        keys = ["mR"] + self.class_names
        values = [recall_dict[key] for key in keys]
        values = ["{:.2f}".format(value * 100) for value in values]
        print("| " + " | ".join(keys) + " |")
        print("|" + " :---: |" * len(keys))
        print("| " + " | ".join(values) + " |")

        metric_dict = dict()
        metric_dict["AP/mAP"] = ap_dict["mAP"]
        metric_dict["Precision/mP"] = precision_dict["mP"]
        metric_dict["Recall/mR"] = recall_dict["mR"]
        for class_name in self.class_names:
            metric_dict[f"AP/{class_name}"] = ap_dict[class_name]
            metric_dict[f"Precision/{class_name}"] = precision_dict[class_name]
            metric_dict[f"Recall/{class_name}"] = recall_dict[class_name]

        return metric_dict

    def _calc_AP(self, prs, rcs):
        """
        calculate AP

        Args:
            prs (np.array): precision array
            rcs (np.array): recall array
        Return:
            AP (float)
            P (float)
            R (float)
        """
        rcs = np.concatenate(([0.0], rcs))
        AP = np.sum((rcs[1:] - rcs[:-1]) * prs)
        P, R = prs[-1], rcs[-1]

        return AP, P, R


def build_metric(cfg):
    """build metric"""
    if cfg["type"] == "ComposeMetric":
        return ComposeMetric(**cfg)
    elif cfg["type"] == "MAPMetric3D":
        return MAPMetric3D(**cfg)
    else:
        raise NotImplementedError
