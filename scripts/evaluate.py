"""Measure counting accuracy against a folder of labelled shelf photos.

Labels: a CSV with rows  image,x1,y1,x2,y2[,class]  (Grocery Dataset / SKU-110K style),
one row per product. Reports MAE, MAPE and share of images within +/-10% of the true count.

  python scripts/evaluate.py --images path/to/ShelfImages --labels annotations.csv --conf 0.2 0.25 0.3
"""
import argparse
import collections
import csv
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.detector import ShelfDetector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--model", default="models/shelfcount.onnx")
    ap.add_argument("--conf", type=float, nargs="+", default=[0.25])
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=0, help="random subset size (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--roi", action="store_true", help="count only detections inside the labelled region")
    ap.add_argument("--only", help="only evaluate images whose file stem exists in this folder (e.g. a val split)")
    a = ap.parse_args()

    rows = [r for r in csv.reader(open(a.labels)) if r and not r[0].startswith("image")]
    truth = collections.Counter(r[0] for r in rows)
    # Region the annotator labelled = bounding box of all its labelled products (+3% margin).
    # Some datasets (e.g. Grocery) only label the central shelf unit; products on neighbouring shelves
    # are real but unlabelled, so --roi counts only detections whose centre falls inside that region.
    region = {}
    for r in rows:
        x1, y1, x2, y2 = map(float, r[1:5])
        g = region.get(r[0], [x1, y1, x2, y2])
        region[r[0]] = [min(g[0], x1), min(g[1], y1), max(g[2], x2), max(g[3], y2)]
    names = sorted(n for n in truth if (Path(a.images) / n).exists())
    if a.only:
        keep = {p.stem for p in Path(a.only).iterdir()}
        names = [n for n in names if Path(n).stem in keep]
    if a.n:
        random.Random(a.seed).shuffle(names)
        names = sorted(names[: a.n])
    det = ShelfDetector(a.model, conf=min(a.conf), iou=a.iou)
    preds = {c: [] for c in a.conf}
    gts, times = [], []
    for n in names:
        img = cv2.imread(str(Path(a.images) / n), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)  # dataset labels use raw pixels
        t = time.perf_counter()
        res = det.predict(img, conf=min(a.conf))
        times.append(time.perf_counter() - t)
        boxes = res.boxes
        if a.roi:
            x1, y1, x2, y2 = region[n]
            mx, my = 0.03 * (x2 - x1), 0.03 * (y2 - y1)
            boxes = [b for b in boxes if x1 - mx <= (b.x1 + b.x2) / 2 <= x2 + mx and y1 - my <= (b.y1 + b.y2) / 2 <= y2 + my]
        for c in a.conf:
            preds[c].append(sum(b.score >= c for b in boxes))
        gts.append(truth[n])
    gts = np.array(gts, float)
    print(f"model={Path(a.model).name} roi={a.roi} images={len(gts)} avg_true={gts.mean():.1f} "
          f"avg_time={np.mean(times):.2f}s")
    print(f"{'conf':>6} {'MAE':>7} {'MAPE':>7} {'within10%':>10} {'bias':>7}")
    for c in a.conf:
        p = np.array(preds[c], float)
        err = p - gts
        print(f"{c:>6.2f} {np.abs(err).mean():>7.2f} {np.mean(np.abs(err) / gts) * 100:>6.1f}% "
              f"{np.mean(np.abs(err) / gts <= 0.10) * 100:>9.1f}% {err.mean():>+7.2f}")


if __name__ == "__main__":
    main()
