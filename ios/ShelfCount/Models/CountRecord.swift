import Foundation
import SwiftData
import UIKit

@Model
final class CountRecord {
    @Attribute(.unique) var id: UUID
    var createdAt: Date
    var count: Int
    var ms: Double
    var mode: String            // "device" | "server"
    var imageFile: String       // JPEG in Documents/shelf-images
    var thumbnail: Data?
    var boxesJSON: Data
    var imageWidth: Double      // coordinate space of the boxes
    var imageHeight: Double
    var serverId: String?       // set once the result exists on the dashboard
    var note: String

    init(count: Int, ms: Double, mode: String, imageFile: String, thumbnail: Data?, boxes: [Box],
         imageSize: CGSize, serverId: String?) {
        self.id = UUID()
        self.createdAt = .now
        self.count = count
        self.ms = ms
        self.mode = mode
        self.imageFile = imageFile
        self.thumbnail = thumbnail
        self.boxesJSON = (try? JSONEncoder().encode(boxes)) ?? Data()
        self.imageWidth = imageSize.width
        self.imageHeight = imageSize.height
        self.serverId = serverId
        self.note = ""
    }

    var boxes: [Box] { (try? JSONDecoder().decode([Box].self, from: boxesJSON)) ?? [] }
    var isSynced: Bool { serverId != nil }
}

enum ImageStore {
    static var dir: URL {
        let d = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("shelf-images")
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    static func save(_ image: UIImage) -> String {
        let name = UUID().uuidString + ".jpg"
        try? image.resized(maxSide: 2048).jpegData(compressionQuality: 0.85)?.write(to: dir.appendingPathComponent(name))
        return name
    }

    static func load(_ name: String) -> UIImage? { UIImage(contentsOfFile: dir.appendingPathComponent(name).path) }

    static func delete(_ name: String) { try? FileManager.default.removeItem(at: dir.appendingPathComponent(name)) }

    static func thumbnail(_ image: UIImage) -> Data? { image.resized(maxSide: 240).jpegData(compressionQuality: 0.7) }
}

enum CSVExporter {
    static func file(for records: [CountRecord]) -> URL {
        var csv = "id,created_at,product_count,processing_ms,mode,synced_server_id\n"
        let f = ISO8601DateFormatter()
        for r in records {
            csv += "\(r.id.uuidString),\(f.string(from: r.createdAt)),\(r.count),\(String(format: "%.0f", r.ms)),\(r.mode),\(r.serverId ?? "")\n"
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("shelfcount-export.csv")
        try? csv.write(to: url, atomically: true, encoding: .utf8)
        return url
    }
}
