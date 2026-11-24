"""Nested leave-planogram-out cross-validation for ShelfCount (paper Sections III-VI).

For each of 4 outer folds (3 planograms each):
  * inner validation layout = the remaining planogram whose photo count is closest to 30
  * train YOLO11n on the other 8 planograms: 15 epochs, mosaic off for the last 3 (12 + 3)
  * export to ONNX, then run the server engine (server/detector.py) at a low threshold on the
    inner-validation and test photos (full-size originals), saving every box score and centre
Zero-shot baselines (YOLOE-11s, text prompts and prompt-free) run once on every photo.
A box kept by greedy NMS at a low threshold and scoring >= tau is exactly a box kept by NMS at tau,
so counts for any tau are computed offline from the saved scores (scripts/analyze_cv.py).

Resumable: finished folds are skipped.
  python scripts/cross_validate.py --src datasets/grocery --out experiments/cv
"""
import argparse
import collections
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from server.detector import ShelfDetector  # noqa: E402

FOLDS = {"F1": ["P05", "P08", "P12"], "F2": ["P01", "P06", "P09"],
         "F3": ["P04", "P07", "P11"], "F4": ["P02", "P03", "P10"]}
PROMPTS = ["product", "bottle", "box", "can", "jar", "packet", "carton", "bag"]
LOW_CONF = 0.05
# The Grocery Dataset's labels are in raw pixel coordinates. 23 of its photos carry EXIF orientation tags
# (180 or 90 degrees) that do not match those labels, so dataset photos are read WITHOUT applying EXIF.
# (Phone uploads in the app and server do apply EXIF; that is the right behaviour for new photos.)
READ = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION


def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0]); ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def build_dataset(src: Path, out: Path, max_side=1280):
    """Quality checks (Section IV.C) + one-class YOLO labels for every usable photo."""
    data = out / "data"
    boxes = collections.defaultdict(list)
    for r in csv.reader(open(src / "annotations.csv")):
        boxes[r[0]].append(tuple(map(float, r[1:5])))
    log, usable = [], {}
    for n, bb in sorted(boxes.items()):
        img = cv2.imread(str(src / "ShelfImages" / n), READ)
        if img is None:
            log.append({"image": n, "check": "1 file opens", "detail": "unreadable", "action": "excluded"}); continue
        h, w = img.shape[:2]
        bad_order = [b for b in bb if not (b[0] < b[2] and b[1] < b[3])]
        outside = [b for b in bb if b[0] < -1 or b[1] < -1 or b[2] > w + 1 or b[3] > h + 1]
        dups = sum(1 for i in range(len(bb)) for j in range(i + 1, len(bb)) if iou(bb[i], bb[j]) > 0.9)
        if bad_order:
            log.append({"image": n, "check": "2 corner order", "detail": f"{len(bad_order)} boxes", "action": "excluded"})
        if outside:
            log.append({"image": n, "check": "3 inside photo", "detail": f"{len(outside)} of {len(bb)} boxes outside {w}x{h}",
                        "action": "excluded"})
        if dups:
            log.append({"image": n, "check": "4 duplicates", "detail": f"{dups} pairs IoU>0.9", "action": "kept, logged"})
        if bad_order or outside:
            continue
        usable[n] = {"w": w, "h": h, "boxes": bb, "planogram": n.split("_")[1]}
        stem = Path(n).stem
        (data / "images").mkdir(parents=True, exist_ok=True); (data / "labels").mkdir(parents=True, exist_ok=True)
        dst = data / "images" / f"{stem}.jpg"
        if not dst.exists():
            s = min(1.0, max_side / max(h, w))
            small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
            cv2.imwrite(str(dst), small, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(data / "labels" / f"{stem}.txt", "w") as f:
            for x1, y1, x2, y2 in bb:
                f.write(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
    json.dump(log, open(out / "quality_log.json", "w"), indent=1)
    return usable, log


def plan_folds(usable):
    per = collections.Counter(v["planogram"] for v in usable.values())
    plan = {}
    for k, test in FOLDS.items():
        rest = sorted(p for p in per if p not in test)
        inner = min(rest, key=lambda p: (abs(per[p] - 30), p))
        plan[k] = {"test": test, "inner_val": [inner], "train": [p for p in rest if p != inner]}
    return plan


def photo_list(usable, planograms):
    return sorted(n for n, v in usable.items() if v["planogram"] in planograms)


def run_engine(det, src, names, usable):
    out = {}
    for n in names:
        img = cv2.imread(str(src / "ShelfImages" / n), READ)
        res = det.predict(img, conf=LOW_CONF)
        out[n] = {"true": len(usable[n]["boxes"]), "ms": res.ms,
                  "boxes": [[b.score, (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2] for b in res.boxes]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--close-mosaic", type=int, default=3)
    ap.add_argument("--skip-zero-shot", action="store_true")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    usable, log = build_dataset(src, out)
    plan = plan_folds(usable)
    json.dump({"folds": plan, "photos": len(usable), "excluded": [l["image"] for l in log if l["action"] == "excluded"]},
              open(out / "folds.json", "w"), indent=1)
    regions = {n: [min(b[0] for b in v["boxes"]), min(b[1] for b in v["boxes"]),
                   max(b[2] for b in v["boxes"]), max(b[3] for b in v["boxes"])] for n, v in usable.items()}
    json.dump(regions, open(out / "labelled_regions.json", "w"))
    print(f"usable photos {len(usable)}; excluded {[l['image'] for l in log if l['action'] == 'excluded']}", flush=True)

    from ultralytics import YOLO
    data_dir = out / "data"
    for k, p in plan.items():
        pred_file = out / f"preds_{k}.json"
        if pred_file.exists():
            print(k, "done, skipping", flush=True); continue
        t0 = time.time()
        fold_dir = out / k; fold_dir.mkdir(exist_ok=True)
        for split, pl in (("train", p["train"]), ("val", p["inner_val"])):
            (fold_dir / f"{split}.txt").write_text(
                "\n".join(str(data_dir / "images" / f"{Path(n).stem}.jpg") for n in photo_list(usable, pl)) + "\n")
        (fold_dir / "data.yaml").write_text(f"train: {fold_dir / 'train.txt'}\nval: {fold_dir / 'val.txt'}\nnames:\n  0: product\n")
        model = YOLO("yolo11n.pt")
        model.train(data=str(fold_dir / "data.yaml"), imgsz=640, epochs=a.epochs, close_mosaic=a.close_mosaic, batch=8,
                    device="cpu", workers=2, project=str(fold_dir), name="train", exist_ok=True, single_cls=True,
                    max_det=1500, seed=0, deterministic=True, patience=100, plots=False, verbose=False,
                    fliplr=0.5, flipud=0.0, degrees=3, perspective=0.0005, scale=0.5, hsv_h=0.02, mosaic=1.0)
        best = fold_dir / "train" / "weights" / "best.pt"
        onnx = Path(YOLO(str(best)).export(format="onnx", imgsz=640, simplify=True, dynamic=False, opset=17))
        det = ShelfDetector(onnx, conf=LOW_CONF, iou=0.5)
        preds = {"fold": k, "plan": p, "train_minutes": round((time.time() - t0) / 60, 1),
                 "inner_val": run_engine(det, src, photo_list(usable, p["inner_val"]), usable),
                 "test": run_engine(det, src, photo_list(usable, p["test"]), usable)}
        json.dump(preds, open(pred_file, "w"))
        print(k, f"finished in {preds['train_minutes']} min", flush=True)

    if not a.skip_zero_shot:
        from ultralytics import YOLOE
        for mode in ("yoloe_text", "yoloe_free"):
            f = out / f"preds_{mode}.json"
            if f.exists():
                continue
            if mode == "yoloe_text":
                m = YOLOE("yoloe-11s-seg.pt"); m.set_classes(PROMPTS, m.get_text_pe(PROMPTS))
            else:
                m = YOLOE("yoloe-11s-seg-pf.pt")
            res = {}
            for n in sorted(usable):
                t = time.perf_counter()
                r = m.predict(cv2.imread(str(src / "ShelfImages" / n), READ), conf=LOW_CONF, iou=0.5, imgsz=1024, agnostic_nms=True,
                              max_det=3000, verbose=False)[0]
                xyxy, sc = r.boxes.xyxy.numpy(), r.boxes.conf.numpy()
                res[n] = {"true": len(usable[n]["boxes"]), "ms": (time.perf_counter() - t) * 1000,
                          "boxes": [[float(s), float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)] for b, s in zip(xyxy, sc)]}
            json.dump(res, open(f, "w"))
            print(mode, "done", flush=True)

    # Speed (Section VI.K): end-to-end from JPEG bytes, median of 5 runs after 1 warm-up, on fold F4's model
    if not (out / "timing.json").exists():
        det = ShelfDetector(next((out / "F4" / "train" / "weights").glob("best.onnx")), conf=0.5, iou=0.5)
        rows = []
        for n in photo_list(usable, plan["F4"]["test"])[:50]:
            raw = (src / "ShelfImages" / n).read_bytes()
            def once():
                t = time.perf_counter()
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), READ)
                t1 = time.perf_counter(); det.predict(img); t2 = time.perf_counter()
                return (t2 - t) * 1000, (t2 - t1) * 1000
            once()
            runs = [once() for _ in range(5)]
            rows.append({"image": n, "end_to_end_ms": float(np.median([r[0] for r in runs])),
                         "engine_ms": float(np.median([r[1] for r in runs]))})
        blob = np.zeros((1, 3, det.imgsz, det.imgsz), np.float32)
        model_ms = []
        for _ in range(21):
            t = time.perf_counter(); det.session.run(None, {det.input_name: blob}); model_ms.append((time.perf_counter() - t) * 1000)
        json.dump({"photos": rows, "model_ms_median": float(np.median(model_ms[1:]))}, open(out / "timing.json", "w"), indent=1)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
