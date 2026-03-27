//
//  ContentView.swift
//  echocancellation
//
//  Created by thanhhigk on 25/3/26.
//

import SwiftUI

struct ContentView: View {
    @StateObject private var vm = AECLabViewModel()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    GroupBox("AEC Experiment Control") {
                        VStack(alignment: .leading, spacing: 12) {
                            Toggle("AEC Enabled", isOn: $vm.aecEnabled)
                                .onChange(of: vm.aecEnabled) { _, _ in
                                    vm.syncLiveAECEnabled()
                                }
                            Picker("Scenario", selection: $vm.selectedScenario) {
                                ForEach(LoopbackScenario.allCases) { scenario in
                                    Text(scenario.title).tag(scenario)
                                }
                            }
                            .pickerStyle(.segmented)

                            HStack(spacing: 12) {
                                Button("Run Selected") { vm.runSelectedScenario() }
                                    .buttonStyle(.borderedProminent)
                                    .disabled(vm.isRunning)
                                Button("Run All") { vm.runAllScenarios() }
                                    .buttonStyle(.bordered)
                                    .disabled(vm.isRunning)
                                Button("Stop") { vm.stop() }
                                    .buttonStyle(.bordered)
                                    .disabled(!vm.isRunning)
                            }
                            Text(vm.progressText)
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                    }

                    GroupBox("Live Test (Mic + Speaker)") {
                        VStack(alignment: .leading, spacing: 12) {
                            HStack(spacing: 12) {
                                Button("Start Live") { vm.startLiveTest() }
                                    .buttonStyle(.borderedProminent)
                                    .disabled(vm.isLiveRunning)
                                Button("Stop Live") { vm.stopLiveTest() }
                                    .buttonStyle(.bordered)
                                    .disabled(!vm.isLiveRunning)
                            }
                            Text(vm.liveStatusText)
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                            Text("Hướng dẫn: bật Start Live, để máy phát âm thanh reference qua loa, nói gần mic để xem ERLE/DT thay đổi realtime.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }

                    GroupBox("Live Metrics") {
                        VStack(alignment: .leading, spacing: 8) {
                            metricRow("ERLE", value: String(format: "%.2f dB", vm.latestERLE))
                            metricRow("Delay", value: String(format: "%.2f ms", vm.latestDelayMs))
                            metricRow("DT Ratio", value: String(format: "%.1f %%", vm.latestDTRatio))
                            metricRow("Filter Norm", value: String(format: "%.3f", vm.latestFilterNorm))
                        }
                    }

                    GroupBox("Diagnostic") {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(vm.lastSummary.isEmpty ? "No summary yet." : vm.lastSummary)
                                .font(.footnote)
                            if !vm.exportPath.isEmpty {
                                Text("CSV: \(vm.exportPath)")
                                    .font(.caption2)
                                    .textSelection(.enabled)
                            }
                        }
                    }

                    GroupBox("Scenario Results") {
                        if vm.results.isEmpty {
                            Text("No results yet.")
                                .foregroundStyle(.secondary)
                        } else {
                            VStack(alignment: .leading, spacing: 10) {
                                ForEach(vm.results) { result in
                                    VStack(alignment: .leading, spacing: 4) {
                                        HStack {
                                            Text(result.scenario.title).font(.headline)
                                            Spacer()
                                            Text(result.passed ? "PASS" : "FAIL")
                                                .fontWeight(.bold)
                                                .foregroundStyle(result.passed ? .green : .red)
                                        }
                                        Text(
                                            String(
                                                format: "ERLE %.2f dB | DT %.1f%% | Ratio %.3f | Proc %.2f ms",
                                                result.erleDB, result.dtRatio * 100, result.residualToMicRatio, result.avgProcessMs
                                            )
                                        )
                                        .font(.footnote)
                                        Text(result.note)
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                    .padding(.vertical, 4)
                                    Divider()
                                }
                            }
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("AEC Lab")
        }
    }

    @ViewBuilder
    private func metricRow(_ title: String, value: String) -> some View {
        HStack {
            Text(title)
            Spacer()
            Text(value).monospacedDigit()
        }
    }
}

#Preview {
    ContentView()
}
