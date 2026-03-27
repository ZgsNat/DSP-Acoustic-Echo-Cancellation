import Foundation
import QuartzCore

final class AECPipeline {
    let cfg: AECConfig
    let diagnosticLogger: DiagnosticLogger?

    private let delayEstimator: DelayEstimator
    private let delayLine: DelayLine
    private let dtd: GeigelDTD
    private let nlms: NLMSFilter
    private let nls: NonlinearSuppressor

    private var micPowerAcc: [Float] = []
    private var outPowerAcc: [Float] = []
    private var dtCount = 0
    private var frameCount = 0
    private var lastRefWasSilence = false

    init(config: AECConfig = AECConfig(), diagnosticLogger: DiagnosticLogger? = nil) {
        self.cfg = config
        self.diagnosticLogger = diagnosticLogger
        self.delayEstimator = DelayEstimator(
            sampleRate: config.sampleRate,
            frameSize: config.frameSize,
            maxDelayMs: config.maxDelayMs
        )
        self.delayLine = DelayLine(maxDelaySamples: 48_000)
        self.dtd = GeigelDTD(
            sampleRate: config.sampleRate,
            threshold: config.dtdThreshold,
            hangoverMs: config.dtdHangoverMs
        )
        self.nlms = NLMSFilter(
            config: NLMSConfig(filterLength: config.filterLength, mu: config.mu, eps: config.eps)
        )
        self.nls = NonlinearSuppressor(
            frameSize: config.frameSize,
            alpha: config.nlsAlpha,
            beta: config.nlsBeta
        )
    }

    func process(micFrame: [Float], refFrame: [Float]) -> [Float] {
        let t0 = CACurrentMediaTime()
        guard micFrame.count == cfg.frameSize, refFrame.count == cfg.frameSize else {
            return micFrame
        }

        micPowerAcc.append(DSPMath.meanPower(micFrame))
        frameCount += 1

        let delay = delayEstimator.update(refFrame: refFrame, micFrame: micFrame)
        let refAligned = delayLine.process(refFrame, delay: delay)
        let isDT = dtd.detect(micFrame: micFrame, refFrame: refAligned)
        if isDT { dtCount += 1 }

        let residual = nlms.process(micFrame: micFrame, refFrame: refAligned, update: !isDT)
        let echoEstimate = zip(micFrame, residual).map { $0 - $1 }
        var clean = nls.process(
            residualFrame: residual,
            echoEstimateFrame: echoEstimate,
            refFrame: refAligned,
            isDoubleTalk: isDT
        )

        // Output should suppress, never amplify compared to mic frame energy.
        let cleanEnergy = DSPMath.meanPower(clean)
        let micEnergy = DSPMath.meanPower(micFrame)
        var nlsBypassed = false
        if cleanEnergy > micEnergy, micEnergy > 1e-10 {
            let scale = sqrt(micEnergy / cleanEnergy)
            clean = clean.map { $0 * scale }
            nlsBypassed = true
        }

        outPowerAcc.append(DSPMath.meanPower(clean))

        if let diagnosticLogger {
            let micRMS = DSPMath.rms(micFrame)
            let outRMS = DSPMath.rms(clean)
            let erleInst: Float = micRMS > 1e-6 ? 10 * log10((micRMS * micRMS) / (outRMS * outRMS + 1e-10)) : 0
            let m = FrameMetrics(
                frameIndex: frameCount,
                timestamp: Date().timeIntervalSince1970,
                micRMS: micRMS,
                micPeak: micFrame.map(abs).max() ?? 0,
                refRMS: DSPMath.rms(refFrame),
                refPeak: refFrame.map(abs).max() ?? 0,
                refIsSilence: lastRefWasSilence,
                delaySamples: delay,
                delayMs: Float(delay) * 1000 / Float(cfg.sampleRate),
                isDoubleTalk: isDT,
                nlmsResidualRMS: DSPMath.rms(residual),
                echoEstimateRMS: DSPMath.rms(echoEstimate),
                nlsOutputRMS: outRMS,
                nlsBypassed: nlsBypassed,
                erleInstantDB: erleInst,
                filterNorm: nlms.weightNorm,
                processTimeMs: Float((CACurrentMediaTime() - t0) * 1000)
            )
            diagnosticLogger.log(m)
        }

        return clean
    }

    func feedReference(_ refFrame: [Float]) {
        let delay = delayEstimator.currentDelaySamples
        let refAligned = delayLine.process(refFrame, delay: delay)
        nlms.feedReference(refAligned)
    }

    func markRefSilence(_ value: Bool) {
        lastRefWasSilence = value
    }

    func getMetrics(reset: Bool = true) -> AECMetricsSnapshot {
        let micPower = max(1e-10, micPowerAcc.reduce(0, +) / Float(max(1, micPowerAcc.count)))
        let outPower = max(1e-10, outPowerAcc.reduce(0, +) / Float(max(1, outPowerAcc.count)))
        let erle = 10 * log10(micPower / outPower)
        let snapshot = AECMetricsSnapshot(
            erleDB: erle,
            doubleTalkRatio: Float(dtCount) / Float(max(1, frameCount)),
            frameCount: frameCount,
            delayMs: delayEstimator.currentDelayMs,
            filterNorm: nlms.weightNorm
        )
        if reset {
            micPowerAcc.removeAll(keepingCapacity: true)
            outPowerAcc.removeAll(keepingCapacity: true)
            dtCount = 0
            frameCount = 0
        }
        return snapshot
    }

    func reset() {
        delayLine.reset()
        nlms.reset()
        dtd.reset()
        nls.reset()
        micPowerAcc.removeAll(keepingCapacity: true)
        outPowerAcc.removeAll(keepingCapacity: true)
        dtCount = 0
        frameCount = 0
    }
}
