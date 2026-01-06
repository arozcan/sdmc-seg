#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import nibabel as nib
from PIL import Image


# -----------------------------
# Helpers
# -----------------------------
def read_dataset_json(root: Path) -> Optional[Dict]:
    dj = root / "dataset.json"
    if not dj.exists():
        return None
    with open(dj, "r", encoding="utf-8") as f:
        return json.load(f)


def find_flair_channel_index(dataset_json: Optional[Dict]) -> Optional[int]:
    """
    Tries to infer FLAIR channel index from dataset.json.
    nnU-Net v2 uses channel_names; older uses modality.
    Returns None if not found.
    """
    if not dataset_json:
        return None

    mapping = None
    if isinstance(dataset_json.get("channel_names"), dict):
        mapping = dataset_json["channel_names"]
    elif isinstance(dataset_json.get("modality"), dict):
        mapping = dataset_json["modality"]

    if not mapping:
        return None

    for k, v in mapping.items():
        if isinstance(v, str) and "flair" in v.lower():
            try:
                return int(k)
            except ValueError:
                pass
    return None


def split_modalities(img_nii: Path) -> np.ndarray:
    """
    Returns modalities as np array shaped (C, H, W, D).
    Handles common layouts: (H,W,D,C) or (C,H,W,D).
    """
    img = nib.load(str(img_nii))
    data = img.get_fdata(dtype=np.float32)

    if data.ndim != 4:
        raise ValueError(f"Expected 4D image (multimodal). Got shape={data.shape} for {img_nii}")

    # channels last -> (H,W,D,C)
    if data.shape[-1] in (4, 3, 2):
        data = np.moveaxis(data, -1, 0)  # -> (C,H,W,D)
        return data

    # channels first -> (C,H,W,D)
    if data.shape[0] in (4, 3, 2):
        return data

    raise ValueError(f"Cannot infer channel axis for shape={data.shape} in {img_nii}")


def compute_volume_window(vol: np.ndarray, p_lo: float, p_hi: float) -> Tuple[float, float]:
    """
    Compute global (volume-level) percentile window for normalization.
    """
    x = vol.astype(np.float32)
    # istersen 0'ları atmak için aşağıyı açabilirsin:
    # x = x[x != 0]
    lo = np.percentile(x, p_lo)
    hi = np.percentile(x, p_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(x.min()), float(x.max())
        if hi <= lo:
            hi = lo + 1.0
    return float(lo), float(hi)


def normalize_slice_to_uint(x2d: np.ndarray, lo: float, hi: float, png16: bool) -> np.ndarray:
    """
    Normalize using fixed [lo, hi] from the whole volume.
    """
    x = x2d.astype(np.float32)
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)

    if png16:
        x = (x * 65535.0).round().astype(np.uint16)
    else:
        x = (x * 255.0).round().astype(np.uint8)
    return x


def rotate_left_90(arr2d: np.ndarray) -> np.ndarray:
    # sola 90 derece: k=1
    return np.rot90(arr2d, k=1)


def case_id_from_nii(path: Path) -> str:
    # BRATS_001.nii.gz -> BRATS_001
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


# -----------------------------
# Slice selection
# -----------------------------
def select_slices(label3d: np.ndarray,
                  max_slices_per_case: int = 6,
                  want_empty_one: bool = True,
                  rng: Optional[np.random.Generator] = None) -> List[int]:
    """
    Select up to max_slices_per_case indices on axial plane (z-axis).
    Strategy:
      - reserve 1 slot for empty if want_empty_one and empty exists
      - pick top-K tumor slices by area
      - if tumor < K, fill near tumor band
      - optionally add 1 empty slice far from tumor band
    """
    if rng is None:
        rng = np.random.default_rng(23)

    D = label3d.shape[2]  # axial slices
    areas = np.array([(label3d[:, :, i] > 0).sum() for i in range(D)], dtype=np.int64)
    tumor_idxs = np.where(areas > 0)[0].tolist()
    empty_idxs = np.where(areas == 0)[0].tolist()

    # no tumor? fallback: pick center slices
    if len(tumor_idxs) == 0:
        # pick evenly spaced slices
        if max_slices_per_case <= 0:
            return []
        lin = np.linspace(0, D - 1, num=max_slices_per_case, dtype=int).tolist()
        lin = sorted(set(lin))
        # if want empty, it is already empty anyway
        return lin[:max_slices_per_case]

    # reserve 1 for empty if possible
    reserve_empty = 1 if (want_empty_one and len(empty_idxs) > 0 and max_slices_per_case >= 2) else 0
    K_tumor = max_slices_per_case - reserve_empty

    # top tumor slices by area
    tumor_pairs = [(i, int(areas[i])) for i in tumor_idxs]
    tumor_pairs.sort(key=lambda x: x[1], reverse=True)
    selected = [i for i, _ in tumor_pairs[:K_tumor]]

    # fill if not enough tumor slices
    if len(selected) < K_tumor:
        zmin, zmax = min(tumor_idxs), max(tumor_idxs)
        # fill from margins around tumor band (near context)
        margin = 10
        candidates = []
        for j in range(zmin - margin, zmax + margin + 1):
            if 0 <= j < D and j not in selected:
                candidates.append(j)
        need = K_tumor - len(selected)
        selected += candidates[:need]

    selected = sorted(set(selected))

    # add 1 empty slice closest to the middle slice of the volume
    # (user request: when picking a non-tumor slice, choose the one nearest the center slice)
    if reserve_empty == 1 and len(selected) < max_slices_per_case:
        center_z = 0.5 * (D - 1)
        empty_pick = min(empty_idxs, key=lambda z: abs(z - center_z))
        selected.append(int(empty_pick))

    selected = sorted(set(selected))

    # enforce max
    if len(selected) > max_slices_per_case:
        # keep tumor-heavy ones first: sort by (is_tumor, area)
        def score(z):
            a = int(areas[z])
            return (1 if a > 0 else 0, a)

        selected.sort(key=score, reverse=True)
        selected = selected[:max_slices_per_case]
        selected = sorted(selected)

    return selected


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="Task01_BrainTumour root (contains imagesTr/, labelsTr/, dataset.json)")
    ap.add_argument("--out", type=str, required=True,
                    help="Output root (will create imagesTr/, labelsTr/, dataset.json here)")
    ap.add_argument("--split", type=str, default="Tr", choices=["Tr", "Ts"],
                    help="Use imagesTr or imagesTs (labels expected only for Tr)")
    ap.add_argument("--max_slices_per_case", type=int, default=6,
                    help="Total slices to export per case (includes 1 empty slice if available).")
    ap.add_argument("--include_empty_one", action="store_true",
                    help="If set, adds 1 empty slice per case when available.")
    ap.add_argument("--p_lo", type=float, default=0.5, help="Lower percentile for volume window.")
    ap.add_argument("--p_hi", type=float, default=99.5, help="Upper percentile for volume window.")
    ap.add_argument("--png16", action="store_true",
                    help="Save image slices as 16-bit PNG (recommended for training). If not set, saves 8-bit.")
    ap.add_argument("--flair_index", type=int, default=-1,
                    help="Override flair channel index. If -1, infer from dataset.json, else fallback=0.")
    ap.add_argument("--prefix", type=str, default="brats",
                    help="Filename prefix for exported 2D cases.")
    ap.add_argument("--test_ratio", type=float, default=0.2,
                    help="Fraction of cases to reserve as test split (e.g., 0.2 = 20%). Applied only when split=Tr.")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    images_dir = root / f"images{args.split}"
    labels_dir = root / f"labels{args.split}"
    has_labels = (args.split == "Tr") and labels_dir.exists()

    img_files = sorted(images_dir.glob("*.nii.gz"))

    # --- Train/Test split (case-level) ---
    if args.split == "Tr" and args.test_ratio > 0:
        rng_cases = np.random.default_rng(23)
        img_files = list(img_files)
        rng_cases.shuffle(img_files)
        n_test = int(round(len(img_files) * args.test_ratio))
        test_cases = set(img_files[:n_test])
        train_cases = img_files[n_test:]
    else:
        test_cases = set()
        train_cases = list(img_files)

    out_images_tr = out_root / "imagesTr"
    out_labels_tr = out_root / "labelsTr"
    out_images_ts = out_root / "imagesTs"
    out_labels_ts = out_root / "labelsTs"

    for p in [out_images_tr, out_labels_tr, out_images_ts, out_labels_ts]:
        p.mkdir(parents=True, exist_ok=True)

    dataset_json = read_dataset_json(root)
    inferred = find_flair_channel_index(dataset_json)

    if args.flair_index >= 0:
        flair_idx = args.flair_index
    elif inferred is not None:
        flair_idx = inferred
    else:
        flair_idx = 0  # common fallback

    rng = np.random.default_rng(23)

    # track label max to build dataset.json robustly
    global_label_max = 0
    num_exported = 0

    for img_path in train_cases + list(test_cases):
        cid = case_id_from_nii(img_path)
        is_test = img_path in test_cases
        img_out_dir = out_images_ts if is_test else out_images_tr
        lbl_out_dir = out_labels_ts if is_test else out_labels_tr

        modalities = split_modalities(img_path)  # (C,H,W,D)
        if flair_idx < 0 or flair_idx >= modalities.shape[0]:
            raise RuntimeError(f"Invalid flair_idx={flair_idx} for {img_path}, modalities shape={modalities.shape}")

        flair = modalities[flair_idx]  # (H,W,D)

        # read label if exists
        label = None
        if has_labels:
            lbl_path = labels_dir / img_path.name
            if not lbl_path.exists():
                print(f"⚠️ Label yok, atlandı: {lbl_path}")
                continue
            label = nib.load(str(lbl_path)).get_fdata(dtype=np.float32)
            if label.ndim != 3:
                raise ValueError(f"Expected 3D label. Got shape={label.shape} for {lbl_path}")
            # update max label
            global_label_max = max(global_label_max, int(np.nanmax(label)))

        # volume window for flair
        lo, hi = compute_volume_window(flair, args.p_lo, args.p_hi)

        # choose slices (axial)
        if label is not None:
            selected = select_slices(
                label3d=label,
                max_slices_per_case=args.max_slices_per_case,
                want_empty_one=args.include_empty_one,
                rng=rng,
            )
        else:
            # no labels (Ts): just sample evenly
            D = flair.shape[2]
            k = max(1, args.max_slices_per_case)
            selected = np.linspace(0, D - 1, num=k, dtype=int).tolist()
            selected = sorted(set(selected))[:args.max_slices_per_case]

        # export
        for z in selected:
            x2d = flair[:, :, z]
            x_u = normalize_slice_to_uint(x2d, lo, hi, png16=args.png16)
            x_u = rotate_left_90(x_u)
            img_out_name = f"{args.prefix}_{cid}_{z:03d}_0000.png"
            Image.fromarray(x_u).save(img_out_dir / img_out_name)

            if label is not None and not is_test:
                lbl2d = label[:, :, z].astype(np.uint16 if global_label_max > 255 else np.uint8)
                lbl2d = rotate_left_90(lbl2d)
                lbl_out_name = f"{args.prefix}_{cid}_{z:03d}.png"
                Image.fromarray(lbl2d).save(lbl_out_dir / lbl_out_name)

            num_exported += 1

        print(f"[OK] {cid}: exported {len(selected)} slices (empty_one={args.include_empty_one}, png16={args.png16})")

    # NOTE:
    # Train/Test split is performed at CASE level (patient-wise),
    # not slice-wise, to avoid data leakage.

    # dataset.json (for nnU-Net raw-like layout, 2D PNG)
    # labels: attempt robust mapping (BraTS/MSD Task01 commonly includes 0..3 or 0..4)
    # We'll include up to global_label_max if >0.
    labels_map = {"background": 0}
    if global_label_max >= 1:
        labels_map["edema"] = 1
    if global_label_max >= 2:
        labels_map["non-enhancing tumor"] = 2
    if global_label_max >= 3:
        labels_map["enhancing tumour"] = 3
    if global_label_max >= 4:
        labels_map["others"] = 4

    dj_out = {
        "name": "BrainTumour2D_FLAIR",
        "description": "2D axial slices exported from multimodal Brain Tumour dataset (FLAIR only).",
        "tensorImageSize": "2D",
        "reference": "",
        "licence": "",
        "release": "1.0",
        "channel_names": {"0": "FLAIR"},
        "modality": {"0": "FLAIR"},
        "labels": labels_map,
        "numTraining": len(list(out_images_tr.glob("*.png"))),
        "numTest": len(list(out_images_ts.glob("*.png"))),
        "file_ending": ".png"
    }

    out_path = out_root / "dataset.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dj_out, f, indent=2, ensure_ascii=False)

    print(f"🧾 dataset.json yazıldı: {out_path}")
    print(f"✅ Toplam exported slice: {num_exported}")


if __name__ == "__main__":
    main()