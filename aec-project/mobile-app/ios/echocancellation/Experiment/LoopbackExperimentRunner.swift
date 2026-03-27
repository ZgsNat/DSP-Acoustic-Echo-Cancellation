import Foundation
import QuartzCore

struct ExperimentProgress {
    let scenario: LoopbackScenario
    let frameIndex: Int
    let totalFrames: Int
    let latestMetrics: AECMetricsSnapshot
}

final class LoopbackExperimentRunner {
    private let cfg: AECConfig

    init(config: AECConfig = AECConfig()) {
        self.cfg = config
    }

    func run(
        scenario: LoopbackScenario,
        aecEnabled: Bool,
        diagnosticLogger: DiagnosticLogger?,
        onProgress: ((ExperimentProgress) -> Void)? = nil,
        shouldCancel: (() -> Bool)? = nil
    ) -> ScenarioResult {
        let sc = makeScenarioConfig(scenario)
        let far = ExperimentSignalGenerator.generateSpeechLike(
            durationSec: sc.durationSec,
            sampleRate: cfg.sampleRate,
            pauseRatio: sc.farPauseRatio,
            seed: 1001
        )
        let near = sc.nearEnabled
            ? ExperimentSignalGenerator.generateSpeechLike(
                durationSec: sc.durationSec,
                sampleRate: cfg.sampleRate,
                pauseRatio: sc.nearPauseRatio,
                seed: 2002
            )
            : [Float](repeating: 0, count: far.count)
        let rir = ExperimentSignalGenerator.generateRIR(length: cfg.filterLength, sampleRate: cfg.sampleRate)
        let echo = ExperimentSignalGenerator.convolve(signal: far, rir: rir)

        var rng = LCG(seed: 9999)
        let nSamples = min(far.count, near.count)
        let nFrames = nSamples / cfg.frameSize

        let pipeline = AECPipeline(config: cfg, diagnosticLogger: diagnosticLogger)
        var micPowers: [Float] = []
        var outPowers: [Float] = []
        var processTimesMs: [Float] = []
        var maxFilterNorm: Float = 0

        for frame in 0..<nFrames {
            if shouldCancel?() == true {
                break
            }
            let start = frame * cfg.frameSize
            let end = start + cfg.frameSize

            var mic = [Float](repeating: 0, count: cfg.frameSize)
            for i in 0..<cfg.frameSize {
                let noise = (rng.nextFloat() * 2 - 1) * sc.noiseLevel
                mic[i] = near[start + i] + echo[start + i] + noise
            }
            let ref = Array(far[start..<end])

            let dropped = rng.nextFloat() < sc.refDropRate
            let refFrame = dropped ? [Float](repeating: 0, count: cfg.frameSize) : ref
            pipeline.markRefSilence(dropped)

            let t0 = CACurrentMediaTime()
            let out = aecEnabled ? pipeline.process(micFrame: mic, refFrame: refFrame) : mic
            let elapsed = Float((CACurrentMediaTime() - t0) * 1000)

            processTimesMs.append(elapsed)
            micPowers.append(DSPMath.meanPower(mic))
            outPowers.append(DSPMath.meanPower(out))
            maxFilterNorm = max(maxFilterNorm, pipeline.getMetrics(reset: false).filterNorm)

            if frame % 8 == 0 {
                onProgress?(
                    ExperimentProgress(
                        scenario: scenario,
                        frameIndex: frame + 1,
                        totalFrames: nFrames,
                        latestMetrics: pipeline.getMetrics(reset: false)
                    )
                )
            }
        }

        let metrics = pipeline.getMetrics(reset: true)
        let avgMicPower = max(1e-10, micPowers.reduce(0, +) / Float(max(1, micPowers.count)))
        let avgOutPower = max(1e-10, outPowers.reduce(0, +) / Float(max(1, outPowers.count)))
        let erle = 10 * log10(avgMicPower / avgOutPower)
        let ratio = sqrt(avgOutPower / avgMicPower)
        let finalNorm = metrics.filterNorm
        let avgProcess = processTimesMs.reduce(0, +) / Float(max(1, processTimesMs.count))

        var passed = true
        var note = "OK"
        if erle < 5 {
            passed = false
            note = "ERLE too low (\(String(format: "%.2f", erle)) dB)"
        } else if ratio > 1 {
            passed = false
            note = "Residual > mic (ratio \(String(format: "%.2f", ratio)))"
        } else if maxFilterNorm > 50 {
            passed = false
            note = "Filter divergence (max norm \(String(format: "%.2f", maxFilterNorm)))"
        }

        return ScenarioResult(
            scenario: scenario,
            erleDB: erle,
            dtRatio: metrics.doubleTalkRatio,
            residualToMicRatio: ratio,
            maxFilterNorm: maxFilterNorm,
            finalFilterNorm: finalNorm,
            avgProcessMs: avgProcess,
            passed: passed,
            note: note
        )
    }

    private func makeScenarioConfig(_ scenario: LoopbackScenario) -> ScenarioConfig {
        switch scenario {
        case .echoOnly:
            return ScenarioConfig(
                scenario: .echoOnly, durationSec: 4,
                farPauseRatio: 0, nearPauseRatio: 0,
                nearEnabled: false, noiseLevel: 1e-4, refDropRate: 0
            )
        case .intermittent:
            return ScenarioConfig(
                scenario: .intermittent, durationSec: 4,
                farPauseRatio: 0.3, nearPauseRatio: 0,
                nearEnabled: false, noiseLevel: 1e-4, refDropRate: 0
            )
        case .doubleTalk:
            return ScenarioConfig(
                scenario: .doubleTalk, durationSec: 4,
                farPauseRatio: 0.2, nearPauseRatio: 0.3,
                nearEnabled: true, noiseLevel: 1e-4, refDropRate: 0
            )
        case .refSilence:
            return ScenarioConfig(
                scenario: .refSilence, durationSec: 4,
                farPauseRatio: 0, nearPauseRatio: 0,
                nearEnabled: false, noiseLevel: 1e-4, refDropRate: 0.3
            )
        case .worstCase:
            return ScenarioConfig(
                scenario: .worstCase, durationSec: 5,
                farPauseRatio: 0.2, nearPauseRatio: 0.4,
                nearEnabled: true, noiseLevel: 5e-4, refDropRate: 0.2
            )
        }
    }
}
