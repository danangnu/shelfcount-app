import SwiftUI

struct SettingsView: View {
    @EnvironmentObject private var settings: AppSettings
    @State private var status: String?
    @State private var testing = false

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Picker("Count", selection: Binding(get: { settings.mode }, set: { settings.mode = $0 })) {
                        ForEach(CountMode.allCases) { Text($0.label).tag($0) }
                    }
                    .pickerStyle(.segmented)
                } header: {
                    Text("Where to count")
                } footer: {
                    Text(settings.mode == .device
                         ? "Works offline. Photos never leave the phone unless you send them to the dashboard."
                         : "Photos are uploaded to your ShelfCount server and appear on its web dashboard.")
                }

                Section("Detection") {
                    VStack(alignment: .leading) {
                        HStack {
                            Text("Confidence")
                            Spacer()
                            Text(settings.confidence, format: .number.precision(.fractionLength(2))).monospacedDigit()
                        }
                        Slider(value: $settings.confidence, in: 0.05...0.8, step: 0.01)
                        Text("Lower finds more products but may add false ones.")
                            .font(.footnote).foregroundStyle(.secondary)
                    }
                    if let d = OnDeviceDetector.shared {
                        LabeledContent("On-device engine", value: d.engineDescription)
                    }
                }

                Section {
                    TextField("http://192.168.1.10:8000", text: $settings.serverURL)
                        .keyboardType(.URL).textInputAutocapitalization(.never).autocorrectionDisabled()
                    SecureField("API key (optional)", text: $settings.apiKey)
                    Toggle("Send on-device counts to dashboard", isOn: $settings.autoSync)
                    Button {
                        Task {
                            testing = true
                            defer { testing = false }
                            do { status = try await settings.client.health() } catch { status = error.localizedDescription }
                        }
                    } label: {
                        HStack { Text("Test connection"); if testing { Spacer(); ProgressView() } }
                    }
                } header: {
                    Text("Server")
                } footer: {
                    if let status { Text(status) }
                }

                Section("About") {
                    LabeledContent("Version", value: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "1.0")
                }
            }
            .navigationTitle("Settings")
        }
    }
}
