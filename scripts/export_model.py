"""Export a ShelfCount detector to ONNX for the server and the iPhone app.

  # fine-tuned detector (what ships):
  python scripts/export_model.py --weights runs/shelfcount/grocery-n/weights/best.pt --imgsz 640
  # zero-shot baseline (YOLOE with text prompts baked in; weak on dense shelves, kept for comparison):
  python scripts/export_model.py --weights yoloe-11s-seg.pt --imgsz 1024

Writes models/shelfcount.onnx + models/model.json. The server loads it with ONNX Runtime (no PyTorch);
the iPhone app bundles the same file and runs it with ONNX Runtime + Core ML acceleration.
Optional: --formats onnx coreml  (Core ML export is most reliable on a Mac).
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

DEFAULT_PROMPTS = ["product", "bottle", "box", "can", "jar", "packet", "carton", "bag"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yoloe-11s-seg.pt")
    ap.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--out", default="models")
    ap.add_argument("--formats", nargs="+", default=["onnx"])
    a = ap.parse_args()

    from ultralytics import YOLO, YOLOE

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if "yoloe" in Path(a.weights).name:
        model = YOLOE(a.weights)
        model.set_classes(a.prompts, model.get_text_pe(a.prompts))
        names = a.prompts
    else:
        model = YOLO(a.weights)
        names = list(model.names.values())

    meta = {"source_weights": Path(a.weights).name, "classes": names, "imgsz": a.imgsz}
    for fmt in a.formats:
        kw = dict(format=fmt, imgsz=a.imgsz)
        if fmt == "onnx":
            kw.update(simplify=True, dynamic=False, opset=17)
        if fmt == "coreml":
            kw.update(nms=False, half=True)
        path = Path(model.export(**kw))
        dest = out / ("shelfcount.onnx" if fmt == "onnx" else "ShelfCount.mlpackage")
        if dest.exists():
            shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
        shutil.move(str(path), dest)
        if dest.is_file():
            meta[f"{fmt}_sha256"] = hashlib.sha256(dest.read_bytes()).hexdigest()
        print("exported", dest)
    (out / "model.json").write_text(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
