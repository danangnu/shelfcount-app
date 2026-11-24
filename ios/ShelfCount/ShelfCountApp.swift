import SwiftUI
import SwiftData

enum CountMode: String, CaseIterable, Identifiable {
    case device, server
    var id: String { rawValue }
    var label: String { self == .device ? "On this iPhone" : "On server" }
}

/// User settings, saved in UserDefaults. @Published so every screen updates when a value changes.
final class AppSettings: ObservableObject {
    private let store = UserDefaults.standard

    @Published var mode: CountMode { didSet { store.set(mode.rawValue, forKey: "mode") } }
    @Published var serverURL: String { didSet { store.set(serverURL, forKey: "serverURL") } }
    @Published var apiKey: String { didSet { store.set(apiKey, forKey: "apiKey") } }
    @Published var confidence: Double { didSet { store.set(confidence, forKey: "confidence") } }
    @Published var autoSync: Bool { didSet { store.set(autoSync, forKey: "autoSync") } }

    init() {
        let d = UserDefaults.standard
        mode = CountMode(rawValue: d.string(forKey: "mode") ?? "") ?? .device
        serverURL = d.string(forKey: "serverURL") ?? "http://192.168.1.10:8000"
        apiKey = d.string(forKey: "apiKey") ?? ""
        confidence = d.object(forKey: "confidence") as? Double ?? 0.65
        autoSync = d.bool(forKey: "autoSync")
    }

    var client: ServerClient { ServerClient(baseURL: serverURL, apiKey: apiKey) }
}

@main
struct ShelfCountApp: App {
    @StateObject private var settings = AppSettings()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(settings)
                .tint(.green)
        }
        .modelContainer(for: CountRecord.self)
    }
}

struct RootView: View {
    var body: some View {
        TabView {
            CountView()
                .tabItem { Label("Count", systemImage: "camera.viewfinder") }
            HistoryView()
                .tabItem { Label("History", systemImage: "list.bullet.rectangle") }
            SettingsView()
                .tabItem { Label("Settings", systemImage: "gearshape") }
        }
    }
}
