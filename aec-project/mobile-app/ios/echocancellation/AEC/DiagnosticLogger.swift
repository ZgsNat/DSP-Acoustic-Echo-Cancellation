import Foundation

final class DiagnosticLogger {
    private let url: URL
    private let queue = DispatchQueue(label: "aec.logger.queue")
    private var rows: [FrameMetrics] = []

    init?(fileName: String = "aec_debug.csv") {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        guard let dir else { return nil }
        self.url = dir.appendingPathComponent(fileName)
        writeHeader()
    }

    func log(_ m: FrameMetrics) {
        queue.sync {
            rows.append(m)
            appendCSVLine(m)
        }
    }

    func exportURL() -> URL { url }

    func summary() -> String {
        queue.sync {
            guard !rows.isEmpty else { return "No frames recorded." }
            let n = Float(rows.count)
            let dt = Float(rows.filter(\.isDoubleTalk).count) / n
            let avgProc = rows.map(\.processTimeMs).reduce(0, +) / n
            let avgERLE = rows.map(\.erleInstantDB).reduce(0, +) / n
            let finalDelay = rows.last?.delayMs ?? 0
            return String(
                format: "frames=%d, dt=%.1f%%, avgERLE=%.2fdB, avgProc=%.2fms, finalDelay=%.1fms",
                Int(n), dt * 100, avgERLE, avgProc, finalDelay
            )
        }
    }

    private func writeHeader() {
        let header = [
            "frame_idx", "timestamp",
            "mic_rms", "mic_peak", "ref_rms", "ref_peak", "ref_is_silence",
            "delay_samples", "delay_ms",
            "is_double_talk",
            "nlms_residual_rms", "echo_est_rms",
            "nls_output_rms", "nls_bypassed",
            "erle_instant_db", "filter_norm", "process_time_ms"
        ].joined(separator: ",") + "\n"
        try? header.data(using: .utf8)?.write(to: url, options: .atomic)
    }

    private func appendCSVLine(_ m: FrameMetrics) {
        let line = [
            "\(m.frameIndex)", "\(m.timestamp)",
            "\(m.micRMS)", "\(m.micPeak)", "\(m.refRMS)", "\(m.refPeak)", "\(m.refIsSilence ? 1 : 0)",
            "\(m.delaySamples)", "\(m.delayMs)",
            "\(m.isDoubleTalk ? 1 : 0)",
            "\(m.nlmsResidualRMS)", "\(m.echoEstimateRMS)",
            "\(m.nlsOutputRMS)", "\(m.nlsBypassed ? 1 : 0)",
            "\(m.erleInstantDB)", "\(m.filterNorm)", "\(m.processTimeMs)"
        ].joined(separator: ",") + "\n"

        if let handle = try? FileHandle(forWritingTo: url),
           let data = line.data(using: .utf8) {
            try? handle.seekToEnd()
            try? handle.write(contentsOf: data)
            try? handle.close()
        }
    }
}
