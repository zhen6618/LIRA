import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
from torchsparse.tensor import PointTensor
from torchsparse.utils import *
from scipy.ndimage import label, generate_binary_structure
import numpy as np
from copy import deepcopy

from ops.torchsparse_utils import *

__all__ = ['SPVCNN', 'SConv3d', 'ConvGRU']


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride), spnn.BatchNorm(outc),
            spnn.ReLU(True))

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        stride=stride,
                        transposed=True), spnn.BatchNorm(outc),
            spnn.ReLU(True))

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride), spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=1), spnn.BatchNorm(outc))

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                spnn.BatchNorm(outc)
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class SPVCNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.dropout = kwargs['dropout']

        cr = kwargs.get('cr', 1.0)
        cs = [32, 64, 128, 96, 96]
        cs = [int(cr * x) for x in cs]

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.stem = nn.Sequential(
            spnn.Conv3d(kwargs['in_channels'], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[2], cs[3], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[3] + cs[1], cs[3], ks=3, stride=1,
                              dilation=1),
                ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
            )
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[3], cs[4], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[4] + cs[0], cs[4], ks=3, stride=1,
                              dilation=1),
                ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1),
            )
        ])

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[2]),
                nn.BatchNorm1d(cs[2]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[2], cs[4]),
                nn.BatchNorm1d(cs[4]),
                nn.ReLU(True),
            )
        ])

        self.weight_initialization()

        if self.dropout:
            self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        # x: SparseTensor z: PointTensor
        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.stage1(x1)  # 2倍下采样
        x2 = self.stage2(x1)  # 2倍下采样
        z1 = voxel_to_point(x2, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)

        y3 = point_to_voxel(x2, z1)
        if self.dropout:
            y3.F = self.dropout(y3.F)
        y3 = self.up1[0](y3)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up1[1](y3)

        y4 = self.up2[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up2[1](y4)
        z3 = voxel_to_point(y4, z1)
        z3.F = z3.F + self.point_transforms[1](z1.F)

        return z3.F


class SConv3d(nn.Module):
    def __init__(self, inc, outc, pres, vres, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = spnn.Conv3d(inc,
                               outc,
                               kernel_size=ks,
                               dilation=dilation,
                               stride=stride)
        self.point_transforms = nn.Sequential(
            nn.Linear(inc, outc),
        )
        self.pres = pres
        self.vres = vres

    def forward(self, z):
        x = initial_voxelize(z, self.pres, self.vres)
        x = self.net(x)
        out = voxel_to_point(x, z, nearest=False)
        out.F = out.F + self.point_transforms(z.F)
        return out


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128, pres=1, vres=1):
        super(ConvGRU, self).__init__()
        self.convz = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
        self.convr = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
        self.convq = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)

    def forward(self, h, x):
        '''

        :param h: PintTensor
        :param x: PintTensor
        :return: h.F: Tensor (N, C)
        '''
        hx = PointTensor(torch.cat([h.F, x.F], dim=1), h.C)

        z = torch.sigmoid(self.convz(hx).F)
        r = torch.sigmoid(self.convr(hx).F)
        x.F = torch.cat([r * h.F, x.F], dim=1)
        q = torch.tanh(self.convq(x).F)

        h.F = (1 - z) * h.F + z * q
        return h.F

# def keep_largest_connected_region_3d(coords: torch.Tensor, mask: torch.Tensor, grid_shape: tuple) -> torch.Tensor:
#     """
#     Keep the largest volume region among each connected component, removing other smaller regions.
    
#     Args:
#         coords (torch.Tensor): Voxel coordinates with shape (N, 3), each row is an (x, y, z) coordinate.
#         mask (torch.Tensor): Voxel mask with shape (N,), indicating whether each voxel is valid.
#         grid_shape (tuple): Size of the 3D grid (D, H, W), i.e., dimensions of the voxel grid.
        
#     Returns:
#         torch.Tensor: Mask retaining only the largest volume region among each connected component, 
#                       with shape (N,) and type torch.bool.
#     """
#     # Extract coordinates of valid voxels
#     valid_coords = coords[mask].cpu()
    
#     if valid_coords.size(0) == 0:
#         return torch.zeros(mask.size(0), dtype=mask.dtype, device=mask.device)  # Return all False mask if no valid voxels
    
#     # Create 3D grid initialized with 0
#     grid = np.zeros(grid_shape, dtype=int)  # Initialize with 0, later use labels for connected regions
    
#     # Map valid voxel coordinates to 3D grid, indexing at valid voxel positions
#     z, y, x = valid_coords[:, 0].int().numpy(), valid_coords[:, 1].int().numpy(), valid_coords[:, 2].int().numpy()
#     grid[z, y, x] = 1  # Mark positions of valid voxels as 1 in 3D grid
    
#     # Generate 26-connected structural element (3D)
#     structure = generate_binary_structure(3, 3)  # 26-connectivity
    
#     # Use scipy.ndimage.label to mark connected regions in 3D grid
#     labeled_mask, num_labels = label(grid, structure)
    
#     # Calculate volume (number of voxels) for each connected region
#     region_sizes = np.bincount(labeled_mask.ravel())
    
#     # Find the region with maximum volume (excluding background region with label 0)
#     max_label = region_sizes[1:].argmax() + 1  # `argmax` returns index of largest region, +1 because background is labeled 0
    
#     # Create mask containing only the largest connected region
#     max_area_mask_np = (labeled_mask == max_label)
    
#     # Generate final mask by converting coordinates back to valid voxel positions
#     final_mask = max_area_mask_np[z, y, x]  # Index using original valid voxel coordinates
    
#     # Return boolean tensor with shape (N,)
#     return torch.tensor(final_mask, dtype=mask.dtype, device=mask.device)


def keep_largest_connected_region_3d(coords: torch.Tensor, mask: torch.Tensor, grid_shape: tuple, n: int = 1) -> torch.Tensor:
    """
    Retain the largest volume region among the first n connected regions and remove the other small regions.

    参数:
        coords (torch.Tensor): Voxel coordinates, in the shape of (N, 3), each line represents a (x, y, z) coordinate.
        mask (torch.Tensor): Voxel mask, in the shape of (N,), indicates whether each voxel is valid.
        grid_shape (tuple): The size of the 3D mesh (D, H, W), that is, the dimension of the voxel mesh.
        n (int): The default value of the maximum number of connected regions to be retained is 1.

    返回:
        torch.Tensor: Only retain the mask of the first n largest volume regions, with the shape (N,) and type torch.bool.
    """
    valid_coords = coords[mask].cpu()

    if valid_coords.size(0) == 0:
        return torch.zeros(mask.size(0), dtype=mask.dtype, device=mask.device)  

    grid = np.zeros(grid_shape, dtype=int)  

    z, y, x = valid_coords[:, 0].int().numpy(), valid_coords[:, 1].int().numpy(), valid_coords[:, 2].int().numpy()
    grid[z, y, x] = 1 

    structure = generate_binary_structure(3, 3)  # 26- Connectivity

    labeled_mask, num_labels = label(grid, structure)

    # Calculate the volume (i.e., the number of voxels) of each connected region
    region_sizes = np.bincount(labeled_mask.ravel())

    sorted_labels = np.argsort(region_sizes[1:])[::-1] + 1  

    selected_labels = sorted_labels[:n]

    max_area_mask_np = np.isin(labeled_mask, selected_labels)

    final_mask = max_area_mask_np[z, y, x]  

    return torch.tensor(final_mask, dtype=mask.dtype, device=mask.device)

def keep_connected_region_3d(coords, labels, n):
    vis_coords = coords - coords.min(dim=0)[0]  # > [0, 0, 0]
    vis_scope = vis_coords.max(dim=0)[0].cpu().tolist() 
    vis_scope = [vis_x + 10 for vis_x in vis_scope]
    vis_scope = tuple(vis_scope)

    labels_copy = deepcopy(labels)
    uni_labels = torch.unique(labels)

    for uni in uni_labels:
        if uni == 0:
            continue
        else:
            uni = int(uni.cpu().item())

            mask_3d = labels_copy == uni
            mask_3d = keep_largest_connected_region_3d(vis_coords, mask_3d, vis_scope, n)

            labels_copy[labels_copy == uni] = mask_3d.to(mask_3d.dtype) * uni

    return labels_copy


def calculate_iou(mask_pred, mask_gt):
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    if union == 0:
        return 0.0
    return intersection / union


def calculate_ap(fp, tp, fn):
    # Compute precision and recall
    precision = np.cumsum(tp) / (np.cumsum(tp) + np.cumsum(fp))  # Cumulative precision = cumulative TP/cumulative pred
    recall = np.cumsum(tp) / (np.sum(tp) + np.sum(fn))  # Cumulative recall = cumulative TP/GT (FN is not cumulative; it remains the same across all thresholds)

    precision = np.nan_to_num(precision, nan=0.0)
    recall = np.nan_to_num(recall, nan=0.0)

    # Expand on Recall by adding a starting point (Recall=0) and an ending point (Recall=1).
    recall_for_conv = np.copy(recall)
    recall_for_conv = np.insert(recall_for_conv, 0, 0.)  # Insert Recall=0 at the very beginning
    recall_for_conv = np.append(recall_for_conv, 1.)    # Insert Recall=1 at the end

    # The Precision corresponding to the inserted Recall
    precision_for_conv = np.copy(precision)
    precision_for_conv = np.insert(precision_for_conv, 0, 1.)  # When Recall=0, Precision=1
    precision_for_conv = np.append(precision_for_conv, 0.)    # Precision=0 when Recall=1

    # Calculate the area of the PR curve and integrate it using the trapezoidal rule
    ap = np.trapz(precision_for_conv, recall_for_conv)  # trapezoidal integration
    
    return ap

def evaluate_instance_segmentation(predictions_list, ground_truths_list):
    """
    Calculate the AP50, AP75 and mAP of multiple images
    :param predictions_list: List of predictions for each image
    :param ground_truths_list: List of ground truths for each image
    :param iou_thresholds: List of IoU thresholds for calculating mAP
    :return: AP50, AP75, mAP
    """
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    iou_thresholds = np.round(iou_thresholds, 2)  
    iou_thresholds = np.insert(iou_thresholds, 0, 0.25)

    ap_dict = {}
    for iou_threshold in iou_thresholds:   
        all_fp = []
        all_tp = []
        all_fn = []

        for predictions, ground_truths in zip(predictions_list, ground_truths_list):
            fp = []
            tp = []
            fn = []

            # List to track which predictions are already used
            used_predictions = []

            # For each ground truth instance, find the corresponding prediction
            for gt in ground_truths:
                best_iou = 0
                best_pred_i = None
                for i, pred in enumerate(predictions):
                    if i in used_predictions:
                        continue  # Skip if the prediction has already been matched
                    
                    iou = calculate_iou(pred['mask'], gt['mask'])
                    
                    if iou > best_iou and iou > iou_threshold:
                        best_iou = iou
                        best_pred_i = i

                if best_pred_i is not None:
                    tp.append(1)  # True Positive
                    fp.append(0)  # No False Positive for this match
                    fn.append(0)  # No False Negative for this match
                    used_predictions.append(best_pred_i)  # Mark this prediction as used
                else:
                    tp.append(0)  # No True Positive, no match
                    fp.append(0)  # No False Positive
                    fn.append(1)  # False Negative (GT with no matching prediction)

            # For each prediction, check if it doesn't match any ground truth
            for i, pred in enumerate(predictions):
                if i in used_predictions:
                    continue  # Skip if the prediction has already been matched
                else:
                    tp.append(0)  # No True Positive for this prediction
                    fp.append(1)  # False Positive (Prediction with no matching GT)
                    fn.append(0)  # No False Negative


            # Append to the overall lists
            all_fp.extend(fp)
            all_tp.extend(tp)
            all_fn.extend(fn)

        # Calculate AP for different IoU thresholds
        ap = calculate_ap(all_fp, all_tp, all_fn)
        ap_dict[iou_threshold] = ap

    # Calculate AP50 and AP75 from the AP dictionary
    ap25 = ap_dict.get(0.25, 0)
    ap50 = ap_dict.get(0.5, 0)
    ap75 = ap_dict.get(0.75, 0)

    # Calculate mAP (mean AP over all IoU thresholds)
    mAP = np.mean(list(ap_dict.values())[1:])  #  [0.5:0.95:0.05] (AP)

    return ap25, ap50, ap75, mAP
