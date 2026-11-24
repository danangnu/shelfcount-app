#!/usr/bin/env bash
# Copies the counting model into the app bundle, then generates the Xcode project.
# Runs on a Mac or on a GitHub Actions macOS runner:  bash ios/setup.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ShelfCount/Resources
cp ../models/shelfcount.onnx ShelfCount/Resources/shelfcount.onnx
SIZE=$(python3 -c "import json;print(json.load(open('../models/model.json'))['imgsz'])" 2>/dev/null || echo 640)
sed -i.bak "s/ShelfCountInputSize: \"[0-9]*\"/ShelfCountInputSize: \"$SIZE\"/" project.yml && rm -f project.yml.bak
command -v xcodegen >/dev/null || { echo "Install XcodeGen first: brew install xcodegen"; exit 1; }
xcodegen
echo "Done. Open ShelfCount.xcodeproj, pick your Team under Signing & Capabilities, and run on an iPhone."
