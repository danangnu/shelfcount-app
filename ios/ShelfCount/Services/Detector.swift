import CoreML
import CoreVideo
import Foundation
import UIKit

/// One detected product, in original-image pixel coordinates.
struct Box: Codable, Hashable {
    var x1: Double, y1: Double, x2: Double, y2: Double, score: Double
}

struct CountOutput {
    var count: Int
    var boxes: [Box]
    var ms: Double
    var imageSize: CGSize
    var serverId: String? = nil
}

enum DetectorError: LocalizedError {
    case modelMissing, badImage, badOutput(String)
    var errorDescription: String? {
        switch self {
        case .modelMissing: return "The counting model is missing from the app bundle."
        case .badImage: return "This image could not be read."
        case .badOutput(let s): return "Unexpected model output: \(s)"
        }
    }
}

/// Runs the ShelfCount model on the phone with Apple's Core ML (Neural Engine, GPU or CPU, chosen by iOS).
/// It is exported from the same trained weights as the server's ONNX file (scripts/export_model.py):
/// input "image" = a 640 x 640 RGB image (scaled to 0-1 inside the model),
/// output = [1, 4 + classes, anchors] float array (box centre x, centre y, width, height, then class scores).
final class OnDeviceDetector {
    static let shared = try? OnDeviceDetector()

    private let model: MLModel
    private let inputName: String
    private let outputName: String
    let inputSize: Int
    let engineDescription = "Core ML (Neural Engine when available)"

    init(modelName: String = "ShelfCountModel") throws {
        // Xcode compiles ShelfCountModel.mlpackage into ShelfCountModel.mlmodelc inside the app.
        guard let url = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") else { throw DetectorError.modelMissing }
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let loaded = try MLModel(contentsOf: url, configuration: config)
        let desc = loaded.modelDescription
        guard let input = desc.inputDescriptionsByName.first(where: { $0.value.type == .image }) else {
            throw DetectorError.badOutput("model has no image input")
        }
        guard let output = desc.outputDescriptionsByName.first(where: { $0.value.type == .multiArray }) else {
            throw DetectorError.badOutput("model has no array output")
        }
        model = loaded
        inputName = input.key
        outputName = output.key
        inputSize = input.value.imageConstraint?.pixelsWide ?? 640
    }

    func count(_ image: UIImage, conf: Double, iou: Double = 0.5, maxDet: Int = 1500) throws -> CountOutput {
        let start = CFAbsoluteTimeGetCurrent()
        guard let cg = image.normalizedCGImage() else { throw DetectorError.badImage }
        let w = Double(cg.width), h = Double(cg.height)
        let s = inputSize
        let r = min(Double(s) / w, Double(s) / h)
        let nw = Int((w * r).rounded()), nh = Int((h * r).rounded())
        let padX = (s - nw) / 2, padY = (s - nh) / 2

        let buffer = try Self.letterbox(cg, size: s, drawRect: CGRect(x: padX, y: s - padY - nh, width: nw, height: nh))
        let input = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: buffer)])
        let result = try model.prediction(from: input)
        guard let out = result.featureValue(for: outputName)?.multiArrayValue else { throw DetectorError.badOutput("no output") }
        let shape = out.shape.map { $0.intValue }
        guard shape.count == 3 else { throw DetectorError.badOutput("shape \(shape)") }
        let rows = shape[1], n = shape[2]
        let nc = max(1, rows - 4)  // detect export: 4 box values + class scores
        let st = out.strides.map { $0.intValue }
        let values = Self.floats(out)
        @inline(__always) func v(_ row: Int, _ j: Int) -> Float { values[row * st[1] + j * st[2]] }

        var cands: [Box] = []
        for j in 0..<n {
            var best: Float = 0
            for c in 0..<nc { best = max(best, v(4 + c, j)) }
            if Double(best) < conf { continue }
            let cx = Double(v(0, j)), cy = Double(v(1, j)), bw = Double(v(2, j)), bh = Double(v(3, j))
            let x1 = (cx - bw / 2 - Double(padX)) / r, y1 = (cy - bh / 2 - Double(padY)) / r
            let x2 = (cx + bw / 2 - Double(padX)) / r, y2 = (cy + bh / 2 - Double(padY)) / r
            cands.append(Box(x1: max(0, x1), y1: max(0, y1), x2: min(w, x2), y2: min(h, y2), score: Double(best)))
        }
        let kept = Self.nms(cands, iou: iou, maxDet: maxDet).filter { $0.x2 - $0.x1 >= 2 && $0.y2 - $0.y1 >= 2 }
        let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000
        return CountOutput(count: kept.count, boxes: kept, ms: ms, imageSize: CGSize(width: w, height: h))
    }

    /// Letterbox into a size x size BGRA pixel buffer filled with grey (114), like Ultralytics.
    private static func letterbox(_ cg: CGImage, size s: Int, drawRect: CGRect) throws -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [kCVPixelBufferCGImageCompatibilityKey: true, kCVPixelBufferCGBitmapContextCompatibilityKey: true]
        guard CVPixelBufferCreate(kCFAllocatorDefault, s, s, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else { throw DetectorError.badImage }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buffer), width: s, height: s, bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        else { throw DetectorError.badImage }
        ctx.setFillColor(red: 114 / 255, green: 114 / 255, blue: 114 / 255, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: s, height: s))
        ctx.interpolationQuality = .high
        ctx.draw(cg, in: drawRect)  // Core Graphics origin is bottom-left; drawRect is already flipped
        return buffer
    }

    /// Copies an MLMultiArray into a flat Float array (works for float32, float16 and double outputs).
    private static func floats(_ a: MLMultiArray) -> [Float] {
        let n = a.count
        switch a.dataType {
        case .float32:
            let p = a.dataPointer.bindMemory(to: Float.self, capacity: n)
            return Array(UnsafeBufferPointer(start: p, count: n))
        case .double:
            let p = a.dataPointer.bindMemory(to: Double.self, capacity: n)
            return UnsafeBufferPointer(start: p, count: n).map { Float($0) }
        default:
            return (0..<n).map { a[$0].floatValue }
        }
    }

    static func nms(_ boxes: [Box], iou thr: Double, maxDet: Int) -> [Box] {
        let sorted = boxes.sorted { $0.score > $1.score }
        var keep: [Box] = []
        var removed = [Bool](repeating: false, count: sorted.count)
        for i in sorted.indices where !removed[i] {
            let a = sorted[i]
            keep.append(a)
            if keep.count >= maxDet { break }
            let areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
            for j in (i + 1)..<sorted.count where !removed[j] {
                let b = sorted[j]
                let iw = min(a.x2, b.x2) - max(a.x1, b.x1), ih = min(a.y2, b.y2) - max(a.y1, b.y1)
                if iw <= 0 || ih <= 0 { continue }
                let inter = iw * ih
                let union = areaA + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
                if inter / union > thr { removed[j] = true }
            }
        }
        return keep
    }
}

extension UIImage {
    /// Returns a CGImage with orientation applied (camera photos are often rotated via EXIF).
    func normalizedCGImage() -> CGImage? {
        if imageOrientation == .up, let cg = cgImage { return cg }
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let img = UIGraphicsImageRenderer(size: size, format: format).image { _ in draw(in: CGRect(origin: .zero, size: size)) }
        return img.cgImage
    }

    func resized(maxSide: CGFloat) -> UIImage {
        let s = min(1, maxSide / max(size.width, size.height))
        if s >= 1 && imageOrientation == .up { return self }
        let newSize = CGSize(width: (size.width * s).rounded(), height: (size.height * s).rounded())
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        return UIGraphicsImageRenderer(size: newSize, format: format).image { _ in draw(in: CGRect(origin: .zero, size: newSize)) }
    }
}
