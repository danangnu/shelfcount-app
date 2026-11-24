import Foundation
import UIKit
import OnnxRuntimeBindings

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

/// Runs the ShelfCount ONNX model on the phone (Core ML / Neural Engine when available, CPU otherwise).
/// Expects an Ultralytics-style detect export: input "images" [1,3,S,S], output0 [1, 4+classes, anchors].
final class OnDeviceDetector {
    static let shared = try? OnDeviceDetector()

    private let session: ORTSession
    private let env: ORTEnv
    let inputSize: Int
    let usesCoreML: Bool
    private let inputName: String
    private let outputName: String

    init(modelName: String = "shelfcount") throws {
        guard let path = Bundle.main.path(forResource: modelName, ofType: "onnx") else { throw DetectorError.modelMissing }
        let ortEnv = try ORTEnv(loggingLevel: .warning)
        let options = try ORTSessionOptions()
        try options.setGraphOptimizationLevel(.all)
        var coreML = false
        if ORTIsCoreMLExecutionProviderAvailable() {
            let ml = ORTCoreMLExecutionProviderOptions()
            ml.createMLProgram = true
            ml.onlyAllowStaticInputShapes = true
            do {
                try options.appendCoreMLExecutionProvider(with: ml)
                coreML = true
            } catch {
                coreML = false  // fall back to CPU
            }
        }
        let ortSession = try ORTSession(env: ortEnv, modelPath: path, sessionOptions: options)
        let inputs = try ortSession.inputNames()
        let outputs = try ortSession.outputNames()
        env = ortEnv
        session = ortSession
        usesCoreML = coreML
        inputName = inputs.first ?? "images"
        outputName = outputs.first ?? "output0"
        inputSize = Int(Bundle.main.object(forInfoDictionaryKey: "ShelfCountInputSize") as? String ?? "") ?? 640
    }

    func count(_ image: UIImage, conf: Double, iou: Double = 0.5, maxDet: Int = 1500) throws -> CountOutput {
        let start = CFAbsoluteTimeGetCurrent()
        guard let cg = image.normalizedCGImage() else { throw DetectorError.badImage }
        let w = Double(cg.width), h = Double(cg.height)
        let s = inputSize
        let r = min(Double(s) / w, Double(s) / h)
        let nw = Int((w * r).rounded()), nh = Int((h * r).rounded())
        let padX = (s - nw) / 2, padY = (s - nh) / 2

        // Letterbox into an RGBA buffer filled with grey (114), like Ultralytics.
        var rgba = [UInt8](repeating: 114, count: s * s * 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        rgba.withUnsafeMutableBytes { buf in
            guard let ctx = CGContext(data: buf.baseAddress, width: s, height: s, bitsPerComponent: 8, bytesPerRow: s * 4,
                                      space: cs, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return }
            ctx.interpolationQuality = .high
            // CoreGraphics origin is bottom-left.
            ctx.draw(cg, in: CGRect(x: padX, y: s - padY - nh, width: nw, height: nh))
        }
        // RGBA HWC uint8 -> RGB CHW float32 [0,1]
        let plane = s * s
        var floats = [Float](repeating: 0, count: 3 * plane)
        for i in 0..<plane {
            floats[i] = Float(rgba[i * 4]) / 255
            floats[plane + i] = Float(rgba[i * 4 + 1]) / 255
            floats[2 * plane + i] = Float(rgba[i * 4 + 2]) / 255
        }
        let data = NSMutableData(bytes: &floats, length: floats.count * MemoryLayout<Float>.size)
        let input = try ORTValue(tensorData: data, elementType: .float, shape: [1, 3, NSNumber(value: s), NSNumber(value: s)])
        let outputs = try session.run(withInputs: [inputName: input], outputNames: [outputName], runOptions: nil)
        guard let out = outputs[outputName] else { throw DetectorError.badOutput("no output") }
        let shape = try out.tensorTypeAndShapeInfo().shape.map { $0.intValue }
        guard shape.count == 3 else { throw DetectorError.badOutput("shape \(shape)") }
        let rows = shape[1], n = shape[2]
        let nc = max(1, rows - 4)  // detect export: 4 box values + class scores
        let raw = try out.tensorData() as Data

        var cands: [Box] = []
        raw.withUnsafeBytes { (p: UnsafeRawBufferPointer) in
            let f = p.bindMemory(to: Float.self)
            for j in 0..<n {
                var best: Float = 0
                for c in 0..<nc { best = max(best, f[(4 + c) * n + j]) }
                if Double(best) < conf { continue }
                let cx = Double(f[j]), cy = Double(f[n + j]), bw = Double(f[2 * n + j]), bh = Double(f[3 * n + j])
                let x1 = (cx - bw / 2 - Double(padX)) / r, y1 = (cy - bh / 2 - Double(padY)) / r
                let x2 = (cx + bw / 2 - Double(padX)) / r, y2 = (cy + bh / 2 - Double(padY)) / r
                cands.append(Box(x1: max(0, x1), y1: max(0, y1), x2: min(w, x2), y2: min(h, y2), score: Double(best)))
            }
        }
        let kept = Self.nms(cands, iou: iou, maxDet: maxDet).filter { $0.x2 - $0.x1 >= 2 && $0.y2 - $0.y1 >= 2 }
        let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000
        return CountOutput(count: kept.count, boxes: kept, ms: ms, imageSize: CGSize(width: w, height: h))
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
