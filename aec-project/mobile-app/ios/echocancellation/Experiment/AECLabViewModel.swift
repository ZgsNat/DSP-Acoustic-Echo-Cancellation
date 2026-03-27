import Foundation
import SwiftUI
import Combine

@MainActor
final class AECLabViewModel: ObservableObject {
    @Published var selectedScenario: LoopbackScenario = .echoOnly
    @Published var aecEnabled: Bool = true
    @Published var isRunning: Bool = false
    @Published var progressText: String = "Ready"
    @Published var latestERLE: Float = 0
    @Published var latestDelayMs: Float = 0
    @Published var latestDTRatio: Float = 0
    @Published var latestFilterNorm: Float = 0
    @Published var isLiveRunning: Bool = false
    @Published var liveStatusText: String = "Live idle"
    @Published var results: [ScenarioResult] = []
    @Published var lastSummary: String = ""
    @Published var exportPath: String = ""

    private let runner = LoopbackExperimentRunner(
        config: AECConfig(
            sampleRate: 16_000,
            frameSize: 1_024,
            filterLength: 1_024, // Lighter for interactive lab runs on iOS
            mu: 0.5,
            eps: 1e-6,
            maxDelayMs: 300,
            dtdThreshold: 0.8,
            dtdHangoverMs: 100,
            nlsAlpha: 2.5,
            nlsBeta: 0.005
        )
    )
    private var task: Task<Void, Never>?
    private let liveService = LiveAECTestService()

    init() {
        liveService.onSnapshot = { [weak self] s in
            DispatchQueue.main.async {
                guard let self else { return }
                self.latestERLE = s.erleDB
                self.latestDelayMs = s.delayMs
                self.latestDTRatio = s.dtRatioPercent
                self.latestFilterNorm = s.filterNorm
            }
        }
        liveService.onStatus = { [weak self] text in
            DispatchQueue.main.async {
                guard let self else { return }
                self.liveStatusText = text
            }
        }
        liveService.onError = { [weak self] text in
            DispatchQueue.main.async {
                guard let self else { return }
                self.liveStatusText = "Error: \(text)"
                self.isLiveRunning = false
            }
        }
    }

    func runSelectedScenario() {
        guard !isRunning else { return }
        isRunning = true
        progressText = "Running \(selectedScenario.title)..."
        let scenario = selectedScenario
        let aecEnabled = aecEnabled
        let fileName = "aec_\(scenario.rawValue)_\(Int(Date().timeIntervalSince1970)).csv"
        let runner = self.runner

        task = Task {
            let output = await Task.detached(priority: .userInitiated) { () -> (ScenarioResult, String, String) in
                let logger = DiagnosticLogger(fileName: fileName)
                let result = runner.run(
                    scenario: scenario,
                    aecEnabled: aecEnabled,
                    diagnosticLogger: logger,
                    onProgress: { progress in
                        DispatchQueue.main.async { [weak self] in
                            guard let self else { return }
                            self.latestERLE = progress.latestMetrics.erleDB
                            self.latestDelayMs = progress.latestMetrics.delayMs
                            self.latestDTRatio = progress.latestMetrics.doubleTalkRatio * 100
                            self.latestFilterNorm = progress.latestMetrics.filterNorm
                            self.progressText = "\(progress.scenario.title): \(progress.frameIndex)/\(progress.totalFrames)"
                        }
                    },
                    shouldCancel: { Task.isCancelled }
                )
                let summary = logger?.summary() ?? "No diagnostic summary."
                let path = logger?.exportURL().path ?? ""
                return (result, summary, path)
            }.value

            if Task.isCancelled { return }
            let (result, summary, path) = output
            self.results.insert(result, at: 0)
            self.lastSummary = summary
            self.exportPath = path
            self.latestERLE = result.erleDB
            self.latestDTRatio = result.dtRatio * 100
            self.latestFilterNorm = result.finalFilterNorm
            self.isRunning = false
            self.progressText = "Done: \(result.scenario.title) (\(result.passed ? "PASS" : "FAIL"))"
        }
    }

    func runAllScenarios() {
        guard !isRunning else { return }
        isRunning = true
        progressText = "Running all scenarios..."
        results.removeAll()
        let aecEnabled = aecEnabled
        let runner = self.runner

        task = Task {
            for scenario in LoopbackScenario.allCases {
                if Task.isCancelled { break }
                self.progressText = "Running \(scenario.title)..."
                let fileName = "aec_\(scenario.rawValue)_\(Int(Date().timeIntervalSince1970)).csv"
                let output = await Task.detached(priority: .userInitiated) { () -> (ScenarioResult, String, String) in
                    let logger = DiagnosticLogger(fileName: fileName)
                    let result = runner.run(
                        scenario: scenario,
                        aecEnabled: aecEnabled,
                        diagnosticLogger: logger,
                        onProgress: { progress in
                            DispatchQueue.main.async { [weak self] in
                                guard let self else { return }
                                self.latestERLE = progress.latestMetrics.erleDB
                                self.latestDelayMs = progress.latestMetrics.delayMs
                                self.latestDTRatio = progress.latestMetrics.doubleTalkRatio * 100
                                self.latestFilterNorm = progress.latestMetrics.filterNorm
                                self.progressText = "\(progress.scenario.title): \(progress.frameIndex)/\(progress.totalFrames)"
                            }
                        },
                        shouldCancel: { Task.isCancelled }
                    )
                    let summary = logger?.summary() ?? ""
                    let path = logger?.exportURL().path ?? ""
                    return (result, summary, path)
                }.value
                if Task.isCancelled { break }
                let (result, summary, path) = output
                self.results.insert(result, at: 0)
                self.lastSummary = summary
                self.exportPath = path
                self.latestERLE = result.erleDB
                self.latestDTRatio = result.dtRatio * 100
                self.latestFilterNorm = result.finalFilterNorm
            }

            self.isRunning = false
            self.progressText = "Finished all scenarios."
        }
    }

    func stop() {
        task?.cancel()
        task = nil
        isRunning = false
        progressText = "Stopped"
    }

    func startLiveTest() {
        guard !isLiveRunning else { return }
        isLiveRunning = true
        liveStatusText = "Starting live..."
        liveService.start(aecEnabled: aecEnabled)
    }

    func stopLiveTest() {
        guard isLiveRunning else { return }
        liveService.stop()
        isLiveRunning = false
        liveStatusText = "Live stopped"
    }

    func syncLiveAECEnabled() {
        liveService.updateAECEnabled(aecEnabled)
    }
}
