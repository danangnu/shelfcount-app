# ShelfCount

Take a photo of a store shelf and get back one number: how many products are on it.

- **Web dashboard**: drag and drop photos (one or many). Each count is shown next to its thumbnail, and you can toggle box overlays and export CSV.
- **iPhone app** (SwiftUI): photograph a shelf. The app counts on the phone, offline, or sends the photo to the server. History, box overlay, CSV export, and optional sync to the dashboard.
- **REST API** with OpenAPI docs at `/docs`, so it can be plugged into any audit app.
- **One trained model, two formats:** `models/shelfcount.onnx` for the server and `models/ShelfCountModel.mlpackage` (Core ML) for the iPhone, exported from the same weights. A check on GitHub Actions confirms they give the same output.

```
shelfcount/
├── server/            FastAPI app + counting engine (ONNX Runtime, OpenCV) + dashboard (static/)
├── models/            shelfcount.onnx + model.json  (the deployed detector)
├── ios/               iPhone app source (SwiftUI, XcodeGen project.yml)
├── scripts/           train.py, export_model.py, evaluate.py, prepare_grocery.py
├── Dockerfile, docker-compose.yml
└── requirements.txt   runtime deps (no PyTorch)  |  requirements-train.txt  training deps
```

## 1. Run the server and dashboard

**Docker (recommended)**

```bash
docker compose up --build
# open http://localhost:8000      API docs: http://localhost:8000/docs
```

**Python virtualenv**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Settings (environment variables):

| Variable | Default | Meaning |
| --- | --- | --- |
| `SHELFCOUNT_CONF` | `0.65` | Default detection confidence (the dashboard and app can override it per request) |
| `SHELFCOUNT_IOU` | `0.5` | NMS overlap threshold |
| `SHELFCOUNT_API_KEY` | empty | If set, API calls need header `X-API-Key` |
| `SHELFCOUNT_MODEL` | `models/shelfcount.onnx` | Swap in another exported model |
| `SHELFCOUNT_DATA` | `./data` | SQLite database + stored images |
| `SHELFCOUNT_MAX_MB` | `15` | Upload size limit |

Quick API test:

```bash
curl -F file=@samples/shelf.jpg http://localhost:8000/api/count
# {"id":"…","count":37,"ms":180.4, … ,"boxes":[…]}
```

**Small cloud demo**: any 2 vCPU / 4 GB VPS (Hetzner CX22, DigitalOcean, Lightsail) running `docker compose up -d` behind Caddy or nginx for HTTPS. The iPhone app needs HTTPS for servers that aren't on the local network.

## 2. Build the iPhone app

**Without a Mac (GitHub Actions).** Push this folder to a GitHub repository, with `.github/` at the top level and `main` as the branch. The workflow `.github/workflows/ios-build.yml` then compiles the app for the iPhone simulator on a macOS runner, and separately checks that the Core ML model matches the server's ONNX model. It needs no Apple account or secrets. If it fails, the run's Summary page lists the compile errors and the full log is attached as `ios-build-log`. You can also start it by hand: Actions tab → *iOS build check* → *Run workflow*.

**TestFlight upload (GitHub Actions).** `.github/workflows/ios-testflight.yml` archives a signed Release build and uploads it to App Store Connect. It runs only when started by hand (Actions tab → *iOS TestFlight upload* → *Run workflow*). One-time setup:

1. In the Apple Developer site, register the App ID `com.harianda.shelfcount` (Certificates, Identifiers & Profiles → Identifiers).
2. In App Store Connect, create the app (My Apps → + → New App) with that bundle ID.
3. In App Store Connect → Users and Access → Integrations → App Store Connect API, create a **Team** key with **Admin** access (needed for cloud-managed signing). Download the `.p8` file; Apple lets you download it only once.
4. In the GitHub repository → Settings → Secrets and variables → Actions, add `APPLE_TEAM_ID`, `ASC_KEY_ID`, `ASC_ISSUER_ID` and `ASC_KEY_P8` (the whole `.p8` file text).

The build number is the workflow's run number, so every upload is higher than the last. After upload, Apple processes the build for a few minutes before it appears under TestFlight.

**On a Mac.** You need Xcode 15 or later, and an Apple Developer account for a physical device or the App Store.

```bash
brew install xcodegen
cd ios && bash setup.sh       # compiles models/ShelfCountModel.mlpackage, generates ShelfCount.xcodeproj
open ShelfCount.xcodeproj
```

In Xcode, go to *Signing & Capabilities*, choose your Team, change the bundle id if needed, and run on an iPhone (iOS 17+).
No third-party packages: the app uses Apple's Core ML to run the model.

- **On this iPhone** (default): runs offline with Core ML, which uses the Neural Engine when available. Photos never leave the phone.
- **On server**: uploads to the server set in Settings, e.g. `http://192.168.1.10:8000` on the same Wi-Fi.
- **Send on-device counts to dashboard**: after counting on the phone, sends the photo, count and boxes to the server.

## 3. Model: choice, training, results

**How it counts.** It detects every product and then counts the boxes (YOLO11n, single class `product`, 640 px input). Detection gives the box overlay, which lets a reviewer check the number. It is also the base for brand recognition later: add classes or a second-stage classifier.

**What was tried first.** A zero-shot open-vocabulary detector (YOLOE-11s) needs no training. With text prompts ("product", "box", "bottle" …) it counted within ±10% on 0.6% of photos; without prompts, on 12.4%, because it also counts labels and price tags. So fine-tuning is required.

**Training data used for this POC.** The [Grocery Dataset](https://github.com/gulvarol/grocerydataset) (Varol & Kuzu): 354 real shelf photos with 13,184 labelled products, 12 shelf layouts (planograms).

**Results.** Measured by four-round cross-validation that holds out whole shelf layouts, so each photo is counted by a model that never saw its layout: **95.5% of photos within ±10%** of the labelled count, mean error 0.65 products, 0.12 s per photo on 2 CPU cores. Details in `models/RESULTS.md`. The shipped model is trained on all 354 photos with the same recipe.

```bash
pip install -r requirements-train.txt
python scripts/cross_validate.py --src datasets/grocery --out experiments/cv      # trains 4 folds, about 25 min each on CPU
python scripts/analyze_cv.py --cv experiments/cv --src datasets/grocery --detection
python scripts/export_model.py --weights <run>/weights/last.pt --imgsz 640       # writes models/shelfcount.onnx
```

Note: this dataset's labels ignore the photos' EXIF rotation tags (23 photos have one), so the research scripts read dataset photos with `cv2.IMREAD_IGNORE_ORIENTATION`. The server and app do apply EXIF rotation to uploads, which is right for phone photos.

**Dashboard test.** `node tests/dashboard_e2e.js <output-folder>` drives the dashboard in a real browser (upload, batch, drag-and-drop, viewer, delete, settings, CSV, bad files, paging, phone width) against a running server. Set `SHELF_IMAGES` to the dataset's `ShelfImages` folder.

**Recommended next step before production.** Train the same recipe on [SKU-110K](https://github.com/eg4000/SKU110K_CVPR19): 11,762 densely packed supermarket shelves, all product types. Use a GPU (about 1–2 h on one A10/L4, a few dollars of cloud time):

```bash
python scripts/train.py --data SKU-110K.yaml --model yolo11s.pt --imgsz 1024 --epochs 100 --batch 16 --device 0
```

Then fine-tune on 200–500 labelled photos of the buyer's own stores. The POC model was trained only on tobacco shelves, so expect weaker counts on other categories until this is done.

## 4. Areas for improvement

1. **Broader training data**: SKU-110K, then the customer's stores (biggest accuracy gain).
2. **Tiling for very wide or dense shelves**: run the detector on overlapping tiles and merge, for photos with more than ~150 products.
3. **Brand / SKU recognition**: add a classifier on each detected box (embedding + nearest-neighbour against a product gallery).
4. **Section analytics**: cluster boxes into shelf rows and bays; out-of-stock gaps.
5. **Photo-quality guard**: warn on blur, glare or strong angle before counting.
6. **Accounts and multi-tenant storage** for the SaaS version.

## License note

The training/export scripts use Ultralytics YOLO (AGPL-3.0). Selling a closed-source app or SaaS built on these weights needs an [Ultralytics Enterprise license](https://www.ultralytics.com/license), or a switch to an Apache-2.0 detector (RF-DETR, YOLOX, D-FINE) trained with the same data. The server and app code here don't depend on Ultralytics at runtime: they load any ONNX detector with the same output layout.
