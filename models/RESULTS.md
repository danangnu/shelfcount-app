# Model results

Model: YOLO11n, one class (product), 640 px input, ONNX. Deployed threshold: 0.65 (median of the four thresholds chosen in cross-validation).

## Cross-validated accuracy (the numbers to quote)

Nested leave-planogram-out cross-validation on the Grocery Dataset: 354 photos, 12 shelf layouts, 4 outer folds.
Each photo is counted once, by a model that never saw its shelf layout, at a threshold chosen without test photos.

| Method | MAE (products) | MAPE | Photos within ±10% | Bias |
| --- | --- | --- | --- | --- |
| Fine-tuned YOLO11n | 0.65 [0.31, 1.15] | 1.8% | 95.5% [91.4, 98.5] | +0.19 |
| Same, counting only the labelled shelf unit | 0.42 | 1.0% | 98.3% | -0.06 |
| YOLOE-11s zero-shot, prompt-free | 15.05 | 41.7% | 12.4% | -6.34 |
| YOLOE-11s zero-shot, text prompts | 35.73 | 93.8% | 0.6% | -35.73 |

Detection on test folds: precision 0.99, recall 0.98-1.00, mAP50 0.98-0.995.
Speed, 2 vCPU, no GPU: 64 ms model, 116 ms median end to end from JPEG bytes (95th percentile 141 ms).

Of 16 photos outside ±10%, 12 trace to labels (unlabelled neighbouring shelf units, duplicated labels).

Limits: one product category (tobacco). Retrain on SKU-110K and on the buyer's own shelves before relying on other categories.

Important: the dataset's labels use raw pixel orientation. 23 photos carry EXIF rotation tags that the labels ignore,
so research scripts read dataset photos with cv2.IMREAD_IGNORE_ORIENTATION.

Reproduce: python scripts/cross_validate.py --src datasets/grocery --out experiments/cv
           python scripts/analyze_cv.py --cv experiments/cv --src datasets/grocery --detection

The earlier single-split figure (84% within ±10%) is superseded: its threshold was chosen on test photos and it had the orientation error.
