"""Fine-tune the ShelfCount detector (single class: product).

Examples
  # Quick CPU run on the Grocery Dataset (what the shipped POC model used)
  python scripts/train.py --data datasets/grocery-yolo/data.yaml --model yolo11n.pt --imgsz 640 --epochs 40

  # Recommended production run: SKU-110K (11.7k dense supermarket images) on a GPU
  python scripts/train.py --data SKU-110K.yaml --model yolo11s.pt --imgsz 1024 --epochs 100 --batch 16 --device 0

  # Your own shelves: label ~200-500 photos (one box per product, class 0), then
  python scripts/train.py --data my_shelves/data.yaml --model runs/shelfcount/train/weights/best.pt --epochs 50

Then export for server + iPhone:
  python scripts/export_model.py --weights runs/shelfcount/train/weights/best.pt --imgsz 640
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--model", default="yolo11n.pt")
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--device", default="cpu")
ap.add_argument("--name", default="train")
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--fraction", type=float, default=1.0)
ap.add_argument("--mosaic", type=float, default=1.0, help="set 0 for a final polish run")
ap.add_argument("--lr0", type=float, default=0.01)
a = ap.parse_args()

model = YOLO(a.model)
model.train(data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, device=a.device, workers=a.workers,
            project=str(Path("runs/shelfcount").resolve()), name=a.name, exist_ok=True, single_cls=True, max_det=1500,
            patience=15, close_mosaic=5, fraction=a.fraction, seed=0, deterministic=True,
            # shelf photos: keep orientation, allow scale/colour/perspective changes
            fliplr=0.5, flipud=0.0, degrees=3, perspective=0.0005, scale=0.5, hsv_h=0.02, mosaic=a.mosaic,
            lr0=a.lr0, optimizer="SGD" if a.lr0 != 0.01 else "auto",
            plots=False)
metrics = model.val(data=a.data, imgsz=a.imgsz, device=a.device, max_det=1500, plots=False)
print("mAP50", round(metrics.box.map50, 3), "mAP50-95", round(metrics.box.map, 3))
