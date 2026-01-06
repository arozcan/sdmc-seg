import numpy as np
from PIL import Image
from pathlib import Path

pred_dir = Path("nnunet/nnUNet_results/Dataset002_Stroke/preds_fold0_imagesTs")
s=set()
for f in pred_dir.glob("*.png"):
    a=np.array(Image.open(f))
    if a.ndim==3: a=a[...,0]
    s.update(np.unique(a).tolist())

print("PRED global unique:", sorted(s))