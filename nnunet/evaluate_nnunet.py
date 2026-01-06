#!/usr/bin/env python3
import os, glob, csv, argparse
import numpy as np
import torch
from monai.metrics import HausdorffDistanceMetric
from PIL import Image
from pathlib import Path
from typing import Optional


def read_image_any(path: Path) -> np.ndarray:
    """(H,W,3) uint8 RGB image."""
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return np.array(im, dtype=np.uint8)

def make_overlay(image_rgb: np.ndarray, pred_lab: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay predicted labels on image:
      label 1 -> blue, label 2 -> green
    """
    img = image_rgb.astype(np.float32)

    color = np.zeros_like(img, dtype=np.float32)  # (H,W,3)

    # label 1 -> blue
    m1 = (pred_lab == 1)
    color[m1] = np.array([0, 0, 255], dtype=np.float32)

    # label 2 -> green
    m2 = (pred_lab == 2)
    color[m2] = np.array([0, 255, 0], dtype=np.float32)

    mask = (m1 | m2)
    if mask.any():
        # blend only where prediction is 1 or 2
        img[mask] = (1.0 - alpha) * img[mask] + alpha * color[mask]

    return np.clip(img, 0, 255).astype(np.uint8)

def find_matching_image(img_dir: Path, filename: str) -> Optional[Path]:
    """
    Try to find image with same base name in img_dir.
    Priority: exact same name, then same stem with common extensions.
    """
    p = img_dir / filename
    if p.exists():
        return p

    stem = Path(filename).stem
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        cand = img_dir / f"{stem}_0000{ext}"
        if cand.exists():
            return cand
    return None

def read_label_png(path: Path) -> np.ndarray:
    """(H,W) uint8 label map. If RGB/RGBA -> take channel 0."""
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def nanmean_safe(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or np.all(np.isnan(x)):
        return np.nan
    return float(np.nanmean(x))


def dice_iou_for_label(pred_lab: np.ndarray, gt_lab: np.ndarray, label: int, ignore_empty=True):
    """Return (dice, iou) for a single label with ignore_empty behavior like MONAI."""
    pred = (pred_lab == label)
    gt   = (gt_lab == label)

    gt_sum = int(gt.sum())
    pred_sum = int(pred.sum())

    # ignore_empty=True: if GT empty => NaN
    if gt_sum == 0:
        return (np.nan, np.nan) if ignore_empty else ((1.0, 1.0) if pred_sum == 0 else (0.0, 0.0))

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())

    dice_den = (2 * tp + fp + fn)
    iou_den  = (tp + fp + fn)
    dice = (2 * tp) / dice_den if dice_den > 0 else 0.0
    iou  = tp / iou_den if iou_den > 0 else 0.0
    return float(dice), float(iou)


def sensitivity_specificity_for_label(pred_lab: np.ndarray, gt_lab: np.ndarray, label: int, ignore_empty: bool = True):
    """Pixel-wise sensitivity/specificity for a single label (one-vs-rest).

    Returns (sens, spec). If ignore_empty and GT for this label is empty, returns (nan, nan).
    """
    pred = (pred_lab == label)
    gt = (gt_lab == label)

    gt_sum = int(gt.sum())
    if ignore_empty and gt_sum == 0:
        return (np.nan, np.nan)

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())

    sens_den = (tp + fn)
    spec_den = (tn + fp)

    sens = (tp / sens_den) if sens_den > 0 else (np.nan if ignore_empty else 0.0)
    spec = (tn / spec_den) if spec_den > 0 else (np.nan if ignore_empty else 0.0)

    return float(sens), float(spec)


def hd95_for_label(pred_lab: np.ndarray, gt_lab: np.ndarray, label: int, include_background: bool = True):
    """HD95 for a single label using MONAI HausdorffDistanceMetric.

    Note: returns NaN if both pred and gt are empty for the label.
    """
    # Build one-hot tensors on CPU: (B,C,H,W)
    # Channels: background + requested label (simple 2-channel) to reuse MONAI implementation.
    pred_l = (pred_lab == label).astype(np.uint8)
    gt_l = (gt_lab == label).astype(np.uint8)

    # If both empty, HD is undefined -> NaN
    if int(pred_l.sum()) == 0 and int(gt_l.sum()) == 0:
        return np.nan

    # 2-channel one-hot (bg, fg)
    pred_oh = np.stack([(pred_l == 0).astype(np.uint8), pred_l], axis=0)[None, ...]
    gt_oh = np.stack([(gt_l == 0).astype(np.uint8), gt_l], axis=0)[None, ...]

    pred_t = torch.from_numpy(pred_oh).float()
    gt_t = torch.from_numpy(gt_oh).float()

    hd_kwargs = {
        "include_background": include_background,
        "reduction": "none",
        "get_not_nans": False,
    }
    try:
        if "percentile" in HausdorffDistanceMetric.__init__.__code__.co_varnames:
            hd_kwargs["percentile"] = 95
    except Exception:
        pass

    metric = HausdorffDistanceMetric(**hd_kwargs)
    metric(pred_t, gt_t)
    out = metric.aggregate()  # (B,C)
    metric.reset()

    # foreground is channel 1 in this 2-channel setup
    return float(out[0, 1].item())

def binary_dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Square dilation radius r using pure numpy."""
    if r <= 0:
        return mask.astype(bool)
    h, w = mask.shape
    pad = r
    m = np.pad(mask.astype(np.uint8), ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
    out = np.zeros((h, w), dtype=np.uint8)
    for dy in range(0, 2*r + 1):
        for dx in range(0, 2*r + 1):
            out = np.maximum(out, m[dy:dy+h, dx:dx+w])
    return out.astype(bool)

def binary_erode(mask: np.ndarray, r: int) -> np.ndarray:
    """Square erosion radius r using pure numpy."""
    if r <= 0:
        return mask.astype(bool)

    h, w = mask.shape
    pad = r
    m = np.pad(mask.astype(np.uint8),
               ((pad, pad), (pad, pad)),
               mode="constant",
               constant_values=0)

    out = np.ones((h, w), dtype=np.uint8)
    for dy in range(0, 2 * r + 1):
        for dx in range(0, 2 * r + 1):
            out = np.minimum(out, m[dy:dy + h, dx:dx + w])

    return out.astype(bool)


def tiou_for_label(
    pred_lab: np.ndarray,
    gt_lab: np.ndarray,
    label: int,
    r: int,
    ignore_empty: bool = True,
):
    """
    Morphological tolerant IoU (nnU-Net style):

      gt_d = dilate(gt, r)
      gt_e = erode(gt, r)

      intersection = pred & gt_d
      union        = pred | gt_e

      tIoU = intersection / union

    ignore_empty=True:
      - if GT empty -> NaN
    """

    pred = (pred_lab == label)
    gt   = (gt_lab == label)

    gt_sum = int(gt.sum())
    pred_sum = int(pred.sum())

    if gt_sum == 0:
        return np.nan if ignore_empty else (1.0 if pred_sum == 0 else 0.0)

    gt_d = binary_dilate(gt, r)
    gt_e = binary_erode(gt, r)

    intersection = int(np.logical_and(pred, gt_d).sum())
    union        = int(np.logical_or(pred, gt_e).sum())

    return float(intersection / union) if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--labels", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--tiou_radius", type=int, default=2)
    ap.add_argument("--ignore_empty", action="store_true", default=True)
    ap.add_argument("--out_csv", default="nnunet_metrics_numpy.csv")
    ap.add_argument("--img_dir", default=None, help="Input image folder for overlays (filenames should match pred PNGs)")
    ap.add_argument("--overlay_dir", default=None, help="If set, saves overlay images to this folder")
    ap.add_argument("--overlay_alpha", type=float, default=0.5, help="Overlay alpha blending (0..1)")
    args = ap.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    labels = args.labels
    r = args.tiou_radius
    ignore_empty = args.ignore_empty

    pred_paths = sorted(pred_dir.glob("*.png"))
    if not pred_paths:
        raise FileNotFoundError(f"No PNG in pred_dir: {pred_dir}")

    agg = {lab: {"dice": [], "iou": [], "tiou": [], "sens": [], "spec": [], "hd95": []} for lab in labels}
    agg_mean = {"dice": [], "iou": [], "tiou": [], "sens": [], "spec": [], "hd95": []}

    rows = []

    img_dir = Path(args.img_dir) if args.img_dir else None
    overlay_dir = Path(args.overlay_dir) if args.overlay_dir else None
    alpha = float(args.overlay_alpha)

    if overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)
        if img_dir is None:
            raise ValueError("--overlay_dir requires --img_dir (to read the base images).")
        if not img_dir.exists():
            raise FileNotFoundError(f"img_dir not found: {img_dir}")

    for p in pred_paths:
        g = gt_dir / p.name
        if not g.exists():
            print(f"[WARN] GT not found for {p.name} -> expected {g}, skipping.")
            continue

        pred = read_label_png(p)
        gt = read_label_png(g)

        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch {p.name}: pred{pred.shape} gt{gt.shape}")

        
        # >>> NEW: GT'de 0'dan farklı piksel sayısı
        gt_pos_pixels = int((gt != 0).sum())

        # --- Optional overlay saving ---
        if overlay_dir is not None:
            img_path = find_matching_image(img_dir, p.name)
            if img_path is None:
                print(f"[WARN] Overlay image not found for {p.name} in {img_dir}, skipping overlay.")
            else:
                img = read_image_any(img_path)
                if img.shape[:2] != pred.shape:
                    # pred and image must match H,W; if not, warn + skip
                    print(f"[WARN] Overlay shape mismatch for {p.name}: img{img.shape[:2]} pred{pred.shape}, skipping overlay.")
                else:
                    over = make_overlay(img, pred, alpha=alpha)
                    out_path = overlay_dir / p.name  # save as same filename
                    Image.fromarray(over).save(out_path)

        row = {
            "file": p.name,
            "gt_pos_pixels": gt_pos_pixels
        }
        per_d, per_i, per_t, per_s, per_sp, per_h = [], [], [], [], [], []

        for lab in labels:
            d, i = dice_iou_for_label(pred, gt, lab, ignore_empty=ignore_empty)
            t = tiou_for_label(pred, gt, lab, r=r, ignore_empty=ignore_empty)
            s, sp = sensitivity_specificity_for_label(pred, gt, lab, ignore_empty=ignore_empty)
            h = hd95_for_label(pred, gt, lab, include_background=True)

            row[f"dice_{lab}"] = d
            row[f"iou_{lab}"]  = i
            row[f"tiou_{lab}"] = t
            row[f"sens_{lab}"] = s
            row[f"spec_{lab}"] = sp
            row[f"hd95_{lab}"] = h

            agg[lab]["dice"].append(d)
            agg[lab]["iou"].append(i)
            agg[lab]["tiou"].append(t)
            agg[lab]["sens"].append(s)
            agg[lab]["spec"].append(sp)
            agg[lab]["hd95"].append(h)

            per_d.append(d); per_i.append(i); per_t.append(t)
            per_s.append(s); per_sp.append(sp); per_h.append(h)

        md = nanmean_safe(per_d)
        mi = nanmean_safe(per_i)
        mt = nanmean_safe(per_t)
        ms = nanmean_safe(per_s)
        msp = nanmean_safe(per_sp)
        mh = nanmean_safe(per_h)

        mean_suffix = "_".join(map(str, labels))
        row[f"dice_mean_{mean_suffix}"] = md
        row[f"iou_mean_{mean_suffix}"]  = mi
        row[f"tiou_mean_{mean_suffix}"] = mt
        row[f"sens_mean_{mean_suffix}"] = ms
        row[f"spec_mean_{mean_suffix}"] = msp
        row[f"hd95_mean_{mean_suffix}"] = mh

        agg_mean["dice"].append(md)
        agg_mean["iou"].append(mi)
        agg_mean["tiou"].append(mt)
        agg_mean["sens"].append(ms)
        agg_mean["spec"].append(msp)
        agg_mean["hd95"].append(mh)

        rows.append(row)

    if not rows:
        raise RuntimeError("No matched pred/gt pairs found. Check filenames.")

    fieldnames = ["file", "gt_pos_pixels"]
    for lab in labels:
        fieldnames += [
            f"dice_{lab}", f"iou_{lab}", f"tiou_{lab}",
            f"sens_{lab}", f"spec_{lab}", f"hd95_{lab}",
        ]
    mean_key = "_".join(map(str, labels))
    fieldnames += [
        f"dice_mean_{mean_key}",
        f"iou_mean_{mean_key}",
        f"tiou_mean_{mean_key}",
        f"sens_mean_{mean_key}",
        f"spec_mean_{mean_key}",
        f"hd95_mean_{mean_key}",
    ]

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("\n=== SUMMARY (ignore_empty=True -> NaN for empty GT) ===")
    for lab in labels:
        print(
            f"Label {lab}: "
            f"Dice={nanmean_safe(agg[lab]['dice']):.4f}  "
            f"IoU={nanmean_safe(agg[lab]['iou']):.4f}  "
            f"tIoU(r={r})={nanmean_safe(agg[lab]['tiou']):.4f}  "
            f"Sens={nanmean_safe(agg[lab]['sens']):.4f}  "
            f"Spec={nanmean_safe(agg[lab]['spec']):.4f}  "
            f"HD95={nanmean_safe(agg[lab]['hd95']):.4f}"
        )

    print(
        f"Mean(labels {labels}): "
        f"Dice={nanmean_safe(agg_mean['dice']):.4f}  "
        f"IoU={nanmean_safe(agg_mean['iou']):.4f}  "
        f"tIoU(r={r})={nanmean_safe(agg_mean['tiou']):.4f}  "
        f"Sens={nanmean_safe(agg_mean['sens']):.4f}  "
        f"Spec={nanmean_safe(agg_mean['spec']):.4f}  "
        f"HD95={nanmean_safe(agg_mean['hd95']):.4f}"
    )
    print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()