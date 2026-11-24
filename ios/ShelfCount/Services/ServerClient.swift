import Foundation
import UIKit

/// Talks to the ShelfCount FastAPI server (same API the web dashboard uses).
struct ServerClient {
    var baseURL: String
    var apiKey: String

    struct ServerResult: Decodable {
        let id: String
        let count: Int
        let ms: Double
        let width: Int
        let height: Int
        let boxes: [Box]?
    }

    enum ClientError: LocalizedError {
        case badURL, http(Int, String)
        var errorDescription: String? {
            switch self {
            case .badURL: return "The server address is not valid. Check Settings."
            case .http(let code, let msg): return "Server error \(code): \(msg)"
            }
        }
    }

    private func url(_ path: String) throws -> URL {
        let trimmed = baseURL.trimmingCharacters(in: .whitespaces).trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard let u = URL(string: trimmed + path), u.scheme?.hasPrefix("http") == true else { throw ClientError.badURL }
        return u
    }

    /// Server mode: upload the photo, the server counts and stores it.
    func count(image: UIImage, conf: Double) async throws -> CountOutput {
        let jpeg = image.resized(maxSide: 2048).jpegData(compressionQuality: 0.85) ?? Data()
        var form = MultipartForm()
        form.addFile(name: "file", filename: "iphone-\(Int(Date().timeIntervalSince1970)).jpg", mime: "image/jpeg", data: jpeg)
        form.addField(name: "conf", value: String(format: "%.2f", conf))
        form.addField(name: "source", value: "ios")
        let r: ServerResult = try await post("/api/count", form: form)
        // Server boxes are in the stored (<=2048 px) image's coordinates.
        return CountOutput(count: r.count, boxes: r.boxes ?? [], ms: r.ms,
                           imageSize: CGSize(width: r.width, height: r.height), serverId: r.id)
    }

    /// On-device mode with sync: send the phone's own result so it appears on the dashboard.
    func importResult(image: UIImage, output: CountOutput) async throws -> String {
        let scaled = image.resized(maxSide: 2048)
        let s = scaled.size.width / max(1, output.imageSize.width)
        let boxes = output.boxes.map { Box(x1: $0.x1 * s, y1: $0.y1 * s, x2: $0.x2 * s, y2: $0.y2 * s, score: $0.score) }
        var form = MultipartForm()
        form.addFile(name: "file", filename: "iphone-\(Int(Date().timeIntervalSince1970)).jpg", mime: "image/jpeg",
                     data: scaled.jpegData(compressionQuality: 0.85) ?? Data())
        form.addField(name: "count", value: String(output.count))
        form.addField(name: "boxes", value: String(data: try JSONEncoder().encode(boxes), encoding: .utf8) ?? "[]")
        form.addField(name: "ms", value: String(format: "%.1f", output.ms))
        form.addField(name: "model", value: "on-device")
        form.addField(name: "source", value: "ios")
        let r: ServerResult = try await post("/api/results/import", form: form)
        return r.id
    }

    func health() async throws -> String {
        var req = URLRequest(url: try url("/api/health"), timeoutInterval: 8)
        if !apiKey.isEmpty { req.setValue(apiKey, forHTTPHeaderField: "X-API-Key") }
        let (data, resp) = try await URLSession.shared.data(for: req)
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        guard code == 200 else { throw ClientError.http(code, String(data: data, encoding: .utf8) ?? "") }
        let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        return "Connected · model \(obj?["model"] as? String ?? "?")"
    }

    private func post<T: Decodable>(_ path: String, form: MultipartForm) async throws -> T {
        var req = URLRequest(url: try url(path), timeoutInterval: 60)
        req.httpMethod = "POST"
        req.setValue("multipart/form-data; boundary=\(form.boundary)", forHTTPHeaderField: "Content-Type")
        if !apiKey.isEmpty { req.setValue(apiKey, forHTTPHeaderField: "X-API-Key") }
        let (data, resp) = try await URLSession.shared.upload(for: req, from: form.finalized())
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(code) else {
            let detail = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["detail"]
            throw ClientError.http(code, "\(detail ?? String(data: data, encoding: .utf8) ?? "")")
        }
        return try JSONDecoder().decode(T.self, from: data)
    }
}

struct MultipartForm {
    let boundary = "ShelfCount-\(UUID().uuidString)"
    private var body = Data()

    mutating func addField(name: String, value: String) {
        body.append("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(name)\"\r\n\r\n\(value)\r\n".data(using: .utf8)!)
    }

    mutating func addFile(name: String, filename: String, mime: String, data: Data) {
        body.append("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(name)\"; filename=\"\(filename)\"\r\nContent-Type: \(mime)\r\n\r\n".data(using: .utf8)!)
        body.append(data)
        body.append("\r\n".data(using: .utf8)!)
    }

    func finalized() -> Data {
        var d = body
        d.append("--\(boundary)--\r\n".data(using: .utf8)!)
        return d
    }
}
