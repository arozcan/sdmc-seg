#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np


# -------------------------
# ARGUMENTS
# -------------------------
parser = argparse.ArgumentParser("Stroke → nnU-Net converter")
parser.add_argument("--src", type=str, default="dataset/stroke2021",
                    help="Source dataset root (contains imagesTr, labelsTr, dataset.json)")
parser.add_argument("--dst", type=str, default="nnunet/nnUNet_raw/Dataset002_Stroke",
                    help="nnU-Net DatasetXXX folder")
parser.add_argument("--include_background", action="store_true",
                    help="Include nonstroke slices")
args = parser.parse_args()


SRC = Path(args.src)
DST = Path(args.dst)

SRC_META = SRC / "dataset.json"

imagesTr_src = SRC / "imagesTr"
labelsTr_src = SRC / "labelsTr"
imagesTs_src = SRC / "imagesTs"
labelsTs_src = SRC / "labelsTs"

imagesTr_dst = DST / "imagesTr"
labelsTr_dst = DST / "labelsTr"
imagesTs_dst = DST / "imagesTs"
labelsTs_dst = DST / "labelsTs"

for p in [imagesTr_dst, labelsTr_dst, imagesTs_dst, labelsTs_dst]:
    p.mkdir(parents=True, exist_ok=True)


# -------------------------
# LOAD SOURCE METADATA
# -------------------------
with open(SRC_META, "r", encoding="utf-8") as f:
    src_meta = json.load(f)


# -------------------------
# HELPERS
# -------------------------
def save_image_grayscale(src: Path, dst: Path):
    """For CT images: force 1-channel grayscale."""
    im = Image.open(src)
    im = im.convert("L")
    im.save(dst)


def save_label_preserve_values(src: Path, dst: Path):
    """For label maps: preserve class indices exactly (avoid RGB->L conversion)."""
    im = Image.open(src)
    arr = np.array(im)

    # If RGB/RGBA, many pipelines store labels as grayscale duplicated across channels.
    if arr.ndim == 3:
        arr = arr[..., 0]

    # Common binary label convention: {0,255} -> {0,1}
    u = np.unique(arr)
    if set(u.tolist()).issubset({0, 255}):
        arr = (arr > 0).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)

    Image.fromarray(arr, mode="L").save(dst)


def extract_case_ids(entry_list):
    """
    entry_list: list[dict] with keys like 'image_png'
    returns: set of case_id strings
    """
    ids = set()
    for item in entry_list:
        img = item.get("image_png")
        if img is None:
            continue
        ids.add(Path(img).stem)
    return ids


def build_allowed_case_set(split: str):
    """
    split: 'training' or 'test'
    """
    cases = set()

    cases |= extract_case_ids(src_meta.get(f"{split}_ischemic", []))
    cases |= extract_case_ids(src_meta.get(f"{split}_hemorrhagic", []))

    if args.include_background:
        cases |= extract_case_ids(src_meta.get(f"{split}_nonstroke", []))

    return cases


def process_split(images_src, labels_src, images_dst, labels_dst, allowed_ids, prefix):
    processed = []
    missing = 0

    for fname in sorted(os.listdir(images_src)):
        if not fname.lower().endswith(".png"):
            continue

        case_id = Path(fname).stem
        if case_id not in allowed_ids:
            continue

        src_img = images_src / fname
        src_lbl = labels_src / fname

        if not src_lbl.exists():
            print(f"⚠️ Label yok: {src_lbl}")
            missing += 1
            continue

        case_name = f"{prefix}_{case_id}"
        dst_img = images_dst / f"{case_name}_0000.png"
        dst_lbl = labels_dst / f"{case_name}.png"

        save_image_grayscale(src_img, dst_img)
        save_label_preserve_values(src_lbl, dst_lbl)

        # Sanity: warn if labels are not in the expected set {0,1,2}
        lbl_arr = np.array(Image.open(dst_lbl))
        if lbl_arr.ndim == 3:
            lbl_arr = lbl_arr[..., 0]
        uniq = set(np.unique(lbl_arr).tolist())
        if not uniq.issubset({0, 1, 2}):
            print(f"⚠️ Beklenmeyen label değerleri {case_name}: {sorted(list(uniq))} -> {dst_lbl}")

        processed.append(case_name)

    return processed, missing


# -------------------------
# BUILD SPLITS
# -------------------------
train_allowed = build_allowed_case_set("training")
test_allowed  = build_allowed_case_set("test")

print(f"▶ include_background = {args.include_background}")
print(f"▶ Train slices selected: {len(train_allowed)}")
print(f"▶ Test  slices selected: {len(test_allowed)}")


train_cases, miss_tr = process_split(
    imagesTr_src, labelsTr_src,
    imagesTr_dst, labelsTr_dst,
    train_allowed, prefix="stroke"
)

test_cases, miss_ts = process_split(
    imagesTs_src, labelsTs_src,
    imagesTs_dst, labelsTs_dst,
    test_allowed, prefix="stroke"
)


print(f"✅ Train dönüştürüldü: {len(train_cases)}")
if miss_tr:
    print(f"⚠️ Train eksik label: {miss_tr}")

print(f"✅ Test dönüştürüldü: {len(test_cases)}")
if miss_ts:
    print(f"⚠️ Test eksik label: {miss_ts}")


# -------------------------
# CREATE dataset.json (nnU-Net v2)
# -------------------------
dataset_json = {
    "name": "Stroke",
    "description": "Stroke CT slice segmentation (background / ischemic / hemorrhagic)",
    "tensorImageSize": "2D",
    "reference": "",
    "licence": "",
    "release": "1.0",
    "channel_names": {
        "0": "CT"
    },
    "modality": {   # backward compatibility
        "0": "CT"
    },
    "labels": {
        "background": 0,
        "ischemic": 1,
        "hemorrhagic": 2
    },
    "numTraining": len(train_cases),
    "numTest": len(test_cases),
    "file_ending": ".png"
}

out_json = DST / "dataset.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(dataset_json, f, indent=2, ensure_ascii=False)

print(f"🧾 dataset.json yazıldı → {out_json}")