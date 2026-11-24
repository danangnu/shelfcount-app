import SwiftUI
import SwiftData

struct HistoryView: View {
    @Environment(\.modelContext) private var context
    @Query(sort: \CountRecord.createdAt, order: .reverse) private var records: [CountRecord]

    private var total: Int { records.reduce(0) { $0 + $1.count } }

    var body: some View {
        NavigationStack {
            List {
                if !records.isEmpty {
                    Section {
                        HStack {
                            Stat(title: "Photos", value: "\(records.count)")
                            Divider()
                            Stat(title: "Products", value: total.formatted())
                        }
                    }
                }
                Section {
                    ForEach(records) { r in
                        NavigationLink(value: r) { Row(record: r) }
                    }
                    .onDelete { idx in
                        for i in idx {
                            ImageStore.delete(records[i].imageFile)
                            context.delete(records[i])
                        }
                        try? context.save()
                    }
                }
            }
            .overlay {
                if records.isEmpty {
                    ContentUnavailableView("No counts yet", systemImage: "tray",
                                           description: Text("Counted shelf photos appear here."))
                }
            }
            .navigationTitle("History")
            .navigationDestination(for: CountRecord.self) { ResultDetailView(record: $0) }
            .toolbar {
                if !records.isEmpty {
                    ShareLink(item: CSVExporter.file(for: records)) { Label("Export CSV", systemImage: "square.and.arrow.up") }
                }
            }
        }
    }
}

private struct Stat: View {
    let title: String, value: String
    var body: some View {
        VStack(alignment: .leading) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.title2.bold()).monospacedDigit()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct Row: View {
    let record: CountRecord
    var body: some View {
        HStack(spacing: 12) {
            Group {
                if let d = record.thumbnail, let img = UIImage(data: d) {
                    Image(uiImage: img).resizable().scaledToFill()
                } else {
                    Color(.tertiarySystemFill)
                }
            }
            .frame(width: 64, height: 64)
            .clipShape(RoundedRectangle(cornerRadius: 8))

            VStack(alignment: .leading, spacing: 2) {
                Text("\(record.count)").font(.title2.bold()).monospacedDigit()
                Text(record.createdAt.formatted(date: .abbreviated, time: .shortened))
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 4) {
                Image(systemName: record.mode == "server" ? "server.rack" : "iphone")
                if record.isSynced { Image(systemName: "checkmark.icloud").foregroundStyle(.green) }
            }
            .font(.caption).foregroundStyle(.secondary)
        }
    }
}

struct ResultDetailView: View {
    @EnvironmentObject private var settings: AppSettings
    @Environment(\.modelContext) private var context
    @Bindable var record: CountRecord
    @State private var showBoxes = true
    @State private var syncing = false
    @State private var message: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                HStack(alignment: .firstTextBaseline) {
                    Text("\(record.count)").font(.system(size: 56, weight: .heavy, design: .rounded)).foregroundStyle(.green)
                    Text("products").foregroundStyle(.secondary)
                    Spacer()
                    Toggle("Boxes", isOn: $showBoxes).fixedSize()
                }
                if let img = ImageStore.load(record.imageFile) {
                    OverlayImage(image: img, boxes: showBoxes ? record.boxes : [],
                                 space: CGSize(width: record.imageWidth, height: record.imageHeight))
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                }
                LabeledContent("Counted", value: record.createdAt.formatted(date: .long, time: .shortened))
                LabeledContent("Where", value: record.mode == "server" ? "Server" : "On this iPhone")
                LabeledContent("Processing", value: "\(Int(record.ms)) ms")
                LabeledContent("Dashboard", value: record.isSynced ? "Synced" : "Not synced")
                TextField("Note (store, aisle…)", text: $record.note, axis: .vertical)
                    .textFieldStyle(.roundedBorder)

                if !record.isSynced {
                    Button {
                        Task { await sync() }
                    } label: {
                        Label(syncing ? "Sending…" : "Send to dashboard", systemImage: "icloud.and.arrow.up")
                            .frame(maxWidth: .infinity, minHeight: 44)
                    }
                    .buttonStyle(.bordered)
                    .disabled(syncing)
                }
                if let message { Text(message).font(.footnote).foregroundStyle(.secondary) }
            }
            .padding()
        }
        .navigationTitle("Result")
        .navigationBarTitleDisplayMode(.inline)
    }

    @MainActor
    private func sync() async {
        guard let img = ImageStore.load(record.imageFile) else { return }
        syncing = true
        defer { syncing = false }
        let out = CountOutput(count: record.count, boxes: record.boxes, ms: record.ms,
                              imageSize: CGSize(width: record.imageWidth, height: record.imageHeight))
        do {
            record.serverId = try await settings.client.importResult(image: img, output: out)
            try? context.save()
            message = "Sent to dashboard."
        } catch {
            message = error.localizedDescription
        }
    }
}
