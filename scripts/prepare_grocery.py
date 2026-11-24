"""Convert the Grocery Dataset (Varol & Kuzu, github.com/gulvarol/grocerydataset) to YOLO format.

Single class "product". Split by planogram id (P??) so no shelf appears in both train and val.
  python scripts/prepare_grocery.py --src datasets/grocery --out datasets/grocery-yolo
"""
import argparse
import collections
import csv
import random
from pathlib import Path

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--val", type=float, default=0.15)
ap.add_argument("--max-side", type=int, default=1280)
a = ap.parse_args()

src, out = Path(a.src), Path(a.out)
boxes = collections.defaultdict(list)
for r in csv.reader(open(src / "annotations.csv")):
    boxes[r[0]].append(tuple(map(float, r[1:5])))
groups = sorted({n.split("_")[1] for n in boxes})
random.Random(0).shuffle(groups)
val_groups = set(groups[: max(1, round(len(groups) * a.val))])
for n, bb in boxes.items():
    split = "val" if n.split("_")[1] in val_groups else "train"
    img = cv2.imread(str(src / "ShelfImages" / n), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        continue
    h, w = img.shape[:2]
    s = min(1.0, a.max_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    stem = Path(n).stem
    cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(out / "labels" / split / f"{stem}.txt", "w") as f:
        for x1, y1, x2, y2 in bb:
            f.write(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
(out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: product\n")
print("train", len(list((out / "images/train").glob("*"))), "val", len(list((out / "images/val").glob("*"))),
      "val groups", sorted(val_groups))
