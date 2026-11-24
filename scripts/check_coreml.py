"""Check that the iPhone's Core ML model gives the same output as the server's ONNX model.

Runs on macOS only (Core ML prediction needs Apple's runtime):
  pip install coremltools onnxruntime numpy pillow
  python scripts/check_coreml.py --onnx models/shelfcount.onnx --coreml models/ShelfCountModel.mlpackage

Feeds identical 640 x 640 images to both models and compares the raw output arrays
(box centre, size and confidence for all 8,400 anchor points). Writes a short report and
exits with an error if the two models disagree by more than the tolerances below.
"""
import argparse
import sys

import numpy as np
from PIL import Image

SCORE_TOL = 0.02   # confidence, 0-1
BOX_TOL = 2.0      # pixels at the 640 px input
TAU = 0.65         # deployed threshold


def test_images(size, n=4, seed=0):
    rng = np.random.default_rng(seed)
    imgs = [rng.integers(0, 256, (size, size, 3), dtype=np.uint8)]            # noise
    yy, xx = np.mgrid[0:size, 0:size]
    imgs.append(np.stack([xx * 255 // size, yy * 255 // size, (xx + yy) * 127 // size], -1).astype(np.uint8))  # gradient
    # "shelf-like": rows of coloured rectangles on grey, the kind of structure the detector reacts to
    for k in range(n - 2):
        im = np.full((size, size, 3), 114, np.uint8)
        for row in range(5):
            y0 = 20 + row * 125
            x = 10
            while x < size - 60:
                w = int(rng.integers(40, 70)); h = int(rng.integers(70, 110))
                im[y0 + 110 - h:y0 + 110, x:x + w] = rng.integers(0, 256, 3)
                im[y0 + 110 - h + 10:y0 + 110 - h + 25, x + 5:x + w - 5] = 255 - im[y0 + 100, x + 1]
                x += w + int(rng.integers(3, 9))
            im[y0 + 110:y0 + 118, :] = 60
        imgs.append(im)
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--coreml", required=True)
    a = ap.parse_args()

    import coremltools as ct
    import onnxruntime as ort

    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    size = sess.get_inputs()[0].shape[2]
    ml = ct.models.MLModel(a.coreml, compute_units=ct.ComputeUnit.CPU_ONLY)
    in_name = ml.get_spec().description.input[0].name

    worst_score = worst_box = 0.0
    lines, ok = [], True
    for i, im in enumerate(test_images(size)):
        x = (im.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        o = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]                  # [5, N]
        c = np.asarray(next(iter(ml.predict({in_name: Image.fromarray(im)}).values())))[0]
        if c.shape != o.shape:
            print(f"Shape mismatch: ONNX {o.shape}, Core ML {c.shape}")
            sys.exit(1)
        ds = float(np.abs(o[4:] - c[4:]).max())
        conf_mask = (o[4:].max(0) > 0.25) | (c[4:].max(0) > 0.25)                 # boxes that could matter
        db = float(np.abs(o[:4, conf_mask] - c[:4, conf_mask]).max()) if conf_mask.any() else 0.0
        n_o, n_c = int((o[4:].max(0) >= TAU).sum()), int((c[4:].max(0) >= TAU).sum())
        worst_score, worst_box = max(worst_score, ds), max(worst_box, db)
        lines.append(f"image {i}: max confidence diff {ds:.4f}, max box diff {db:.2f} px, "
                     f"anchors >= {TAU}: ONNX {n_o}, Core ML {n_c}")
        ok &= ds <= SCORE_TOL and db <= BOX_TOL

    print("\n".join(lines))
    print(f"worst confidence diff {worst_score:.4f} (limit {SCORE_TOL}), worst box diff {worst_box:.2f} px (limit {BOX_TOL})")
    print("PASS: Core ML matches ONNX" if ok else "FAIL: Core ML output differs from ONNX")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
