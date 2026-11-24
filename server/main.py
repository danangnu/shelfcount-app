"""ShelfCount API + dashboard.

Run:  uvicorn server.main:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .detector import ShelfDetector, draw_overlay

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("SHELFCOUNT_MODEL", ROOT / "models" / "shelfcount.onnx"))
DATA_DIR = Path(os.getenv("SHELFCOUNT_DATA", ROOT / "data"))
CONF = float(os.getenv("SHELFCOUNT_CONF", "0.65"))
IOU = float(os.getenv("SHELFCOUNT_IOU", "0.5"))
API_KEY = os.getenv("SHELFCOUNT_API_KEY", "")
MAX_BYTES = int(os.getenv("SHELFCOUNT_MAX_MB", "15")) * 1024 * 1024
MAX_SIDE = 2048  # stored image long side
THUMB_SIDE = 320

IMG_DIR, THUMB_DIR = DATA_DIR / "images", DATA_DIR / "thumbs"
DB_PATH = DATA_DIR / "shelfcount.db"
_db_lock = threading.Lock()
_infer_lock = threading.Lock()
state: dict = {}


def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def init_storage():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    with db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS results (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL, filename TEXT, count INTEGER NOT NULL,
            width INTEGER, height INTEGER, ms REAL, conf REAL, model TEXT, source TEXT,
            boxes TEXT, note TEXT)""")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_storage()
    state["det"] = ShelfDetector(MODEL_PATH, conf=CONF, iou=IOU)
    yield


app = FastAPI(title="ShelfCount API", version="1.0.0", lifespan=lifespan,
              description="Count products on a retail shelf photo. One image in, one number out.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def check_key(x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Invalid or missing X-API-Key")


def decode_image(data: bytes) -> np.ndarray:
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"Image larger than {MAX_BYTES // 1024 // 1024} MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(415, "Not a readable image (use JPG, PNG or WEBP)")
    return img


def fit(img: np.ndarray, side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = side / max(h, w)
    return img if s >= 1 else cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def row_to_dict(r: sqlite3.Row, with_boxes: bool = False) -> dict:
    d = {k: r[k] for k in r.keys() if k != "boxes"}
    d["image_url"] = f"/api/results/{r['id']}/image"
    d["thumbnail_url"] = f"/api/results/{r['id']}/thumbnail"
    d["overlay_url"] = f"/api/results/{r['id']}/overlay"
    if with_boxes:
        d["boxes"] = json.loads(r["boxes"] or "[]")
    return d


def save_result(img: np.ndarray, filename: str, count: int, boxes: list[dict], ms: float,
                conf: float, model: str, source: str, note: str | None = None) -> dict:
    rid = uuid.uuid4().hex[:12]
    h, w = img.shape[:2]
    stored = fit(img, MAX_SIDE)
    s = stored.shape[1] / w
    boxes = [{**b, "x1": round(b["x1"] * s, 1), "y1": round(b["y1"] * s, 1),
              "x2": round(b["x2"] * s, 1), "y2": round(b["y2"] * s, 1)} for b in boxes]
    cv2.imwrite(str(IMG_DIR / f"{rid}.jpg"), stored, [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(str(THUMB_DIR / f"{rid}.jpg"), fit(img, THUMB_SIDE), [cv2.IMWRITE_JPEG_QUALITY, 80])
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _db_lock, db() as con:
        con.execute("INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (rid, now, filename, count, stored.shape[1], stored.shape[0], ms, conf, model, source,
                     json.dumps(boxes), note))
        r = con.execute("SELECT * FROM results WHERE id=?", (rid,)).fetchone()
    return row_to_dict(r, with_boxes=True)


def run_count(data: bytes, filename: str, conf: float | None, source: str, save: bool) -> dict:
    img = decode_image(data)
    det: ShelfDetector = state["det"]
    with _infer_lock:
        res = det.predict(img, conf=conf)
    boxes = [b.__dict__ for b in res.boxes]
    used_conf = det.conf if conf is None else conf
    if not save:
        return {"count": res.count, "ms": res.ms, "width": res.width, "height": res.height, "boxes": boxes}
    return save_result(img, filename, res.count, boxes, res.ms, used_conf, det.model_name, source)


# ---------------------------------------------------------------- API
@app.get("/api/health")
def health():
    det: ShelfDetector = state["det"]
    return {"status": "ok", "model": det.model_name, "imgsz": det.imgsz, "classes": det.class_names,
            "providers": det.providers, "default_conf": det.conf, "auth": bool(API_KEY)}


@app.post("/api/count", dependencies=[Depends(check_key)])
async def count(file: UploadFile = File(...), conf: float | None = Form(None, ge=0.01, le=0.99),
                source: str = Form("web"), save: bool = Form(True)):
    """Count products in one shelf photo. Returns the total and (if saved) the history record."""
    return run_count(await file.read(), file.filename or "upload.jpg", conf, source, save)


@app.post("/api/count/batch", dependencies=[Depends(check_key)])
async def count_batch(files: list[UploadFile] = File(...), conf: float | None = Form(None, ge=0.01, le=0.99),
                      source: str = Form("web")):
    out = []
    for f in files[:50]:
        try:
            out.append(run_count(await f.read(), f.filename or "upload.jpg", conf, source, True))
        except HTTPException as e:
            out.append({"filename": f.filename, "error": e.detail})
    return {"results": out, "total_products": sum(r.get("count", 0) for r in out)}


@app.post("/api/results/import", dependencies=[Depends(check_key)])
async def import_result(file: UploadFile = File(...), count: int = Form(..., ge=0), boxes: str = Form("[]"),
                        ms: float = Form(0), model: str = Form("on-device"), source: str = Form("ios")):
    """Store a result counted on the phone (on-device mode) so it appears on the dashboard."""
    img = decode_image(await file.read())
    try:
        parsed = json.loads(boxes)
    except json.JSONDecodeError:
        raise HTTPException(400, "boxes must be JSON")
    return save_result(img, file.filename or "device.jpg", count, parsed, ms, 0, model, source)


@app.get("/api/results", dependencies=[Depends(check_key)])
def list_results(limit: int = Query(50, le=500), offset: int = 0):
    with db() as con:
        rows = con.execute("SELECT * FROM results ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                           (limit, offset)).fetchall()
        agg = con.execute("SELECT COUNT(*) n, COALESCE(SUM(count),0) s, COALESCE(AVG(ms),0) a FROM results").fetchone()
    return {"items": [row_to_dict(r) for r in rows],
            "stats": {"images": agg["n"], "products": agg["s"], "avg_ms": round(agg["a"], 1)}}


def get_row(rid: str) -> sqlite3.Row:
    with db() as con:
        r = con.execute("SELECT * FROM results WHERE id=?", (rid,)).fetchone()
    if not r:
        raise HTTPException(404, "Result not found")
    return r


@app.get("/api/results/{rid}", dependencies=[Depends(check_key)])
def get_result(rid: str):
    return row_to_dict(get_row(rid), with_boxes=True)


@app.delete("/api/results/{rid}", dependencies=[Depends(check_key)])
def delete_result(rid: str):
    get_row(rid)
    with _db_lock, db() as con:
        con.execute("DELETE FROM results WHERE id=?", (rid,))
    for d in (IMG_DIR, THUMB_DIR):
        (d / f"{rid}.jpg").unlink(missing_ok=True)
    return {"deleted": rid}


@app.get("/api/results/{rid}/image")
def image(rid: str):
    get_row(rid)
    return FileResponse(IMG_DIR / f"{rid}.jpg", media_type="image/jpeg")


@app.get("/api/results/{rid}/thumbnail")
def thumbnail(rid: str):
    get_row(rid)
    return FileResponse(THUMB_DIR / f"{rid}.jpg", media_type="image/jpeg")


@app.get("/api/results/{rid}/overlay")
def overlay(rid: str):
    r = get_row(rid)
    img = cv2.imread(str(IMG_DIR / f"{rid}.jpg"))
    ok, buf = cv2.imencode(".jpg", draw_overlay(img, json.loads(r["boxes"] or "[]"), r["count"]),
                           [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), media_type="image/jpeg")


@app.get("/api/export.csv", dependencies=[Depends(check_key)])
def export_csv():
    with db() as con:
        rows = con.execute("SELECT id, created_at, filename, count, width, height, ms, conf, model, source "
                           "FROM results ORDER BY created_at DESC").fetchall()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "created_at_utc", "filename", "product_count", "width", "height", "processing_ms",
                "confidence", "model", "source"])
    w.writerows([tuple(r) for r in rows])
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="shelfcount-{stamp}.csv"'})


app.mount("/", StaticFiles(directory=ROOT / "server" / "static", html=True), name="static")
