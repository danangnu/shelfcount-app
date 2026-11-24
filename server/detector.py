"""Shelf product detector: ONNX Runtime + OpenCV only (no PyTorch at runtime).

Works with any Ultralytics-style ONNX export (detect or segment head):
output0 = [1, 4 + num_classes (+ mask coeffs), num_anchors]. All classes are
merged into one "product" count with class-agnostic NMS.
"""
from __future__ import annotations

import ast
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


@dataclass
class CountResult:
    count: int
    boxes: list[Detection] = field(default_factory=list)
    width: int = 0
    height: int = 0
    ms: float = 0.0


class ShelfDetector:
    def __init__(self, model_path: str | Path, conf: float = 0.65, iou: float = 0.5,
                 max_det: int = 1500, threads: int = 0):
        model_path = Path(model_path)
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        providers = [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(model_path), opts, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.imgsz = int(inp.shape[2])
        meta = self.session.get_modelmeta().custom_metadata_map
        try:
            names = ast.literal_eval(meta.get("names", "{}"))  # "{0: 'product', ...}"
        except Exception:
            names = {}
        sidecar = model_path.with_name("model.json")
        if not names and sidecar.exists():
            names = dict(enumerate(json.loads(sidecar.read_text()).get("classes", [])))
        self.num_classes = len(names) or 1
        self.class_names = [names[k] for k in sorted(names)] if names else ["product"]
        self.model_name = model_path.name
        self.providers = self.session.get_providers()
        self.conf, self.iou, self.max_det = conf, iou, max_det
        self.warmup()

    def warmup(self):
        self.session.run(None, {self.input_name: np.zeros((1, 3, self.imgsz, self.imgsz), np.float32)})

    def _letterbox(self, img: np.ndarray):
        h, w = img.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nh, nw = round(h * r), round(w * r)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, r, left, top

    def predict(self, img_bgr: np.ndarray, conf: float | None = None) -> CountResult:
        t0 = time.perf_counter()
        conf = self.conf if conf is None else conf
        h, w = img_bgr.shape[:2]
        canvas, r, padx, pady = self._letterbox(img_bgr)
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.session.run(None, {self.input_name: blob})[0][0]  # [C, N]
        out = out.T  # [N, C]
        scores = out[:, 4:4 + self.num_classes].max(axis=1)
        keep = scores >= conf
        out, scores = out[keep], scores[keep]
        boxes: list[Detection] = []
        if len(out):
            cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
            x1 = (cx - bw / 2 - padx) / r
            y1 = (cy - bh / 2 - pady) / r
            ww, hh = bw / r, bh / r
            rects = np.stack([x1, y1, ww, hh], 1)
            # top_k=0: no cap before NMS (OpenCV applies top_k to candidates, not to kept boxes)
            idx = cv2.dnn.NMSBoxes(rects.tolist(), scores.tolist(), conf, self.iou)
            idx = np.array(idx).reshape(-1)[: self.max_det]
            for i in idx:
                bx1 = float(np.clip(x1[i], 0, w)); by1 = float(np.clip(y1[i], 0, h))
                bx2 = float(np.clip(x1[i] + ww[i], 0, w)); by2 = float(np.clip(y1[i] + hh[i], 0, h))
                if bx2 - bx1 < 2 or by2 - by1 < 2:
                    continue
                boxes.append(Detection(round(bx1, 1), round(by1, 1), round(bx2, 1), round(by2, 1),
                                       round(float(scores[i]), 3)))
        ms = (time.perf_counter() - t0) * 1000
        return CountResult(count=len(boxes), boxes=boxes, width=w, height=h, ms=round(ms, 1))


def draw_overlay(img_bgr: np.ndarray, boxes: list[dict] | list[Detection], count: int) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    t = max(2, round(min(h, w) / 400))
    for b in boxes:
        b = b if isinstance(b, dict) else b.__dict__
        cv2.rectangle(out, (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"])), (80, 220, 60), t)
    label = f"{count} products"
    fs = max(0.8, min(h, w) / 700)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, t + 1)
    cv2.rectangle(out, (0, 0), (tw + 24, th + 24), (20, 20, 20), -1)
    cv2.putText(out, label, (12, th + 12), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), t + 1, cv2.LINE_AA)
    return out
