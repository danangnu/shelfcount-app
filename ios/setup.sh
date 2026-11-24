#!/usr/bin/env bash
# Compiles the Core ML counting model and generates the Xcode project.
# Runs on a Mac or on a GitHub Actions macOS runner:  bash ios/setup.sh
set -euo pipefail
cd "$(dirname "$0")"
MODEL=../models/ShelfCountModel.mlpackage
[ -d "$MODEL" ] || { echo "Missing $MODEL (export it with scripts/export_model.py --formats onnx coreml)"; exit 1; }
rm -rf Generated && mkdir -p Generated
xcrun coremlcompiler compile "$MODEL" Generated/      # writes Generated/ShelfCountModel.mlmodelc
[ -d Generated/ShelfCountModel.mlmodelc ] || { echo "Core ML compile did not produce ShelfCountModel.mlmodelc"; exit 1; }
command -v xcodegen >/dev/null || { echo "Install XcodeGen first: brew install xcodegen"; exit 1; }
xcodegen
echo "Done. Open ShelfCount.xcodeproj, pick your Team under Signing & Capabilities, and run on an iPhone."
