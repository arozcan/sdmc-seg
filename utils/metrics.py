import os
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch import autocast
from monai.data import list_data_collate
from torch.utils.data import DataLoader
import torch.nn.functional as F
from monai.metrics import HausdorffDistanceMetric

def _nanmean_safe(vals):
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size == 0 or np.all(np.isnan(vals)):
        return np.nan
    return float(np.nanmean(vals))

def _dice_iou_onehot(
    pred_onehot: torch.Tensor,   # (B,C,H,W) {0,1}
    gt_onehot: torch.Tensor,     # (B,C,H,W) {0,1}
    c: int,
    ignore_empty: bool = True,
):
    """
    Returns:
        dice: (B,) tensor
        iou : (B,) tensor
    """
    pred = pred_onehot[:, c].bool()
    gt   = gt_onehot[:, c].bool()

    tp = (pred & gt).sum(dim=(1, 2)).float()
    fp = (pred & ~gt).sum(dim=(1, 2)).float()
    fn = (~pred & gt).sum(dim=(1, 2)).float()

    gt_sum = gt.sum(dim=(1, 2))

    dice_den = 2 * tp + fp + fn
    iou_den  = tp + fp + fn

    dice = torch.where(
        dice_den > 0,
        2 * tp / dice_den,
        torch.nan,
    )

    iou = torch.where(
        iou_den > 0,
        tp / iou_den,
        torch.nan,
    )

    if ignore_empty:
        empty = gt_sum == 0
        dice = torch.where(empty, torch.nan, dice)
        iou  = torch.where(empty, torch.nan, iou)
    else:
        empty = gt_sum == 0
        dice = torch.where(empty & (tp == 0), torch.ones_like(dice), dice)
        iou  = torch.where(empty & (tp == 0), torch.ones_like(iou),  iou)

    return dice, iou   # (B,)

def compute_tolerant_iou(
    y_pred: torch.Tensor,   # (B,C,H,W) {0,1} or prob
    y: torch.Tensor,        # (B,C,H,W) {0,1}
    include_background: bool = True,
    ignore_empty: bool = True,
    kernel_size: int = 5,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Returns:
        tIoU: (B,C)
    """

    if y.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_pred {y_pred.shape} vs y {y.shape}")

    if not include_background:
        y_pred = y_pred[:, 1:]
        y = y[:, 1:]

    B, C, H, W = y.shape
    device = y.device

    pred = (y_pred > threshold)
    gt   = (y > 0.5)

    pad = kernel_size // 2
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device)

    def morph(x, mode):
        out = []
        for c in range(x.shape[1]):
            xc = x[:, c:c+1].float()
            conv = F.conv2d(xc, kernel, padding=pad)
            if mode == "dilate":
                out.append(conv > 0)
            elif mode == "erode":
                out.append(conv == kernel.numel())
            else:
                raise ValueError
        return torch.cat(out, dim=1)

    gt_d = morph(gt, "dilate")
    gt_e = morph(gt, "erode")

    intersection = (pred & gt_d).sum(dim=(2, 3)).float()
    union        = (pred | gt_e).sum(dim=(2, 3)).float()

    tiou = torch.where(
        union > 0,
        intersection / union,
        torch.nan,
    )

    if ignore_empty:
        gt_sum = gt.sum(dim=(2, 3))
        tiou = torch.where(gt_sum > 0, tiou, torch.nan)

    return tiou   # (B,C)


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    """Safe division that returns NaN when denominator is zero."""
    return torch.where(
        den > 0,
        num / den,
        torch.tensor(float("nan"), device=num.device, dtype=torch.float32),
    )


def sensitivity_specificity_onehot(
    pred_onehot: torch.Tensor,   # (B,C,H,W) {0,1}
    gt_onehot: torch.Tensor,     # (B,C,H,W) {0,1}
    c: int,
    ignore_empty: bool = True,
):
    """Compute sensitivity and specificity for a single class (one-vs-rest).

    Returns:
        sens: (B,) tensor
        spec: (B,) tensor
    """
    p = pred_onehot[:, c].bool()
    g = gt_onehot[:, c].bool()

    tp = (p & g).sum(dim=(1, 2)).float()
    fp = (p & ~g).sum(dim=(1, 2)).float()
    fn = (~p & g).sum(dim=(1, 2)).float()
    tn = (~p & ~g).sum(dim=(1, 2)).float()

    sens = _safe_div(tp, tp + fn)
    spec = _safe_div(tn, tn + fp)

    if ignore_empty:
        gsum = g.sum(dim=(1, 2))
        empty = gsum == 0
        sens = torch.where(empty, torch.nan, sens)
        spec = torch.where(empty, torch.nan, spec)

    return sens, spec


def hd95_onehot(
    pred_onehot: torch.Tensor,   # (B,C,H,W) {0,1}
    gt_onehot: torch.Tensor,     # (B,C,H,W) {0,1}
    include_background: bool = True,
):
    """Compute HD95 using MONAI HausdorffDistanceMetric.

    Returns:
        hd95: (B,C) tensor if include_background else (B,C-1)

    Notes:
        - HD95 is the 95th percentile Hausdorff distance.
        - This function is intended for evaluation; it expects binary one-hot masks.
    """
    hd_kwargs = {
        "include_background": include_background,
        "reduction": "none",
        "get_not_nans": False,
    }
    # Prefer percentile=95 when available.
    try:
        if "percentile" in HausdorffDistanceMetric.__init__.__code__.co_varnames:
            hd_kwargs["percentile"] = 95
    except Exception:
        pass

    metric = HausdorffDistanceMetric(**hd_kwargs)
    metric(pred_onehot, gt_onehot)
    out = metric.aggregate()
    metric.reset()
    return out