import SwiftUI
import SwiftData
import PhotosUI

struct CountView: View {
    @EnvironmentObject private var settings: AppSettings
    @Environment(\.modelContext) private var context

    @State private var showCamera = false
    @State private var pickerItems: [PhotosPickerItem] = []
    @State private var busy = false
    @State private var progress = ""
    @State private var lastImage: UIImage?
    @State private var lastRecord: CountRecord?
    @State private var errorText: String?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    if let record = lastRecord, let img = lastImage {
                        ResultCard(record: record, image: img)
                    } else {
                        EmptyHint()
                    }

                    if busy {
                        HStack(spacing: 10) { ProgressView(); Text(progress).foregroundStyle(.secondary) }
                            .padding(.vertical, 8)
                    }

                    VStack(spacing: 12) {
                        Button {
                            showCamera = true
                        } label: {
                            Label("Take shelf photo", systemImage: "camera.fill")
                                .frame(maxWidth: .infinity, minHeight: 52)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(busy || !UIImagePickerController.isSourceTypeAvailable(.camera))

                        PhotosPicker(selection: $pickerItems, maxSelectionCount: 20, matching: .images) {
                            Label("Choose from Photos", systemImage: "photo.on.rectangle")
                                .frame(maxWidth: .infinity, minHeight: 52)
                        }
                        .buttonStyle(.bordered)
                        .disabled(busy)
                    }

                    Text("Counting \(settings.mode.label.lowercased())")
                        .font(.footnote).foregroundStyle(.secondary)
                }
                .padding()
            }
            .navigationTitle("ShelfCount")
            .fullScreenCover(isPresented: $showCamera) {
                CameraPicker { image in Task { await process([image]) } }
                    .ignoresSafeArea()
            }
            .onChange(of: pickerItems) { _, items in
                guard !items.isEmpty else { return }
                Task {
                    var images: [UIImage] = []
                    for item in items {
                        if let data = try? await item.loadTransferable(type: Data.self), let img = UIImage(data: data) {
                            images.append(img)
                        }
                    }
                    pickerItems = []
                    await process(images)
                }
            }
            .alert("Could not count", isPresented: .constant(errorText != nil)) {
                Button("OK") { errorText = nil }
            } message: { Text(errorText ?? "") }
        }
    }

    @MainActor
    private func process(_ images: [UIImage]) async {
        guard !images.isEmpty else { return }
        busy = true
        defer { busy = false; progress = "" }
        for (i, image) in images.enumerated() {
            progress = images.count > 1 ? "Counting \(i + 1) of \(images.count)…" : "Counting…"
            do {
                let output = try await runCount(image)
                let record = CountRecord(count: output.count, ms: output.ms, mode: settings.mode.rawValue,
                                         imageFile: ImageStore.save(image), thumbnail: ImageStore.thumbnail(image),
                                         boxes: output.boxes, imageSize: output.imageSize, serverId: output.serverId)
                context.insert(record)
                try? context.save()
                lastImage = image
                lastRecord = record
                UINotificationFeedbackGenerator().notificationOccurred(.success)
                if settings.mode == .device && settings.autoSync {
                    Task { await sync(record, image: image, output: output) }
                }
            } catch {
                errorText = error.localizedDescription
                UINotificationFeedbackGenerator().notificationOccurred(.error)
            }
        }
    }

    private func runCount(_ image: UIImage) async throws -> CountOutput {
        let conf = settings.confidence
        switch settings.mode {
        case .server:
            return try await settings.client.count(image: image, conf: conf)
        case .device:
            guard let detector = OnDeviceDetector.shared else { throw DetectorError.modelMissing }
            return try await Task.detached(priority: .userInitiated) {
                try detector.count(image, conf: conf)
            }.value
        }
    }

    @MainActor
    private func sync(_ record: CountRecord, image: UIImage, output: CountOutput) async {
        if let id = try? await settings.client.importResult(image: image, output: output) {
            record.serverId = id
            try? context.save()
        }
    }
}

private struct EmptyHint: View {
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "square.stack.3d.up")
                .font(.system(size: 54)).foregroundStyle(.green)
            Text("Photograph a shelf").font(.title3.bold())
            Text("Hold the phone level, fit the whole shelf section in frame, and avoid strong glare.")
                .multilineTextAlignment(.center).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 40)
        .background(RoundedRectangle(cornerRadius: 16).fill(Color(.secondarySystemBackground)))
    }
}

struct ResultCard: View {
    let record: CountRecord
    let image: UIImage
    @State private var showBoxes = true

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                Text("\(record.count)")
                    .font(.system(size: 64, weight: .heavy, design: .rounded))
                    .foregroundStyle(.green)
                    .contentTransition(.numericText())
                Text("products").font(.title3).foregroundStyle(.secondary)
                Spacer()
                Toggle("Boxes", isOn: $showBoxes).labelsHidden()
            }
            OverlayImage(image: image, boxes: showBoxes ? record.boxes : [],
                         space: CGSize(width: record.imageWidth, height: record.imageHeight))
                .clipShape(RoundedRectangle(cornerRadius: 12))
            Text("\(record.mode == "server" ? "Server" : "On-device") · \(Int(record.ms)) ms")
                .font(.footnote).foregroundStyle(.secondary)
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 16).fill(Color(.secondarySystemBackground)))
    }
}

/// Draws detection boxes on top of the photo. `space` is the pixel size the boxes are expressed in.
struct OverlayImage: View {
    let image: UIImage
    let boxes: [Box]
    let space: CGSize

    var body: some View {
        Image(uiImage: image)
            .resizable()
            .scaledToFit()
            .overlay {
                GeometryReader { geo in
                    Canvas { ctx, size in
                        let sx = size.width / max(1, space.width), sy = size.height / max(1, space.height)
                        for b in boxes {
                            let rect = CGRect(x: b.x1 * sx, y: b.y1 * sy, width: (b.x2 - b.x1) * sx, height: (b.y2 - b.y1) * sy)
                            ctx.stroke(Path(roundedRect: rect, cornerRadius: 2), with: .color(.green), lineWidth: 2)
                        }
                    }
                    .frame(width: geo.size.width, height: geo.size.height)
                }
            }
    }
}
