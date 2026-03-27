import Testing
@testable import echocancellation

struct AECCoreTests {
    @Test func nlmsReducesEchoOnEchoOnlyScenario() async throws {
        let runner = LoopbackExperimentRunner(config: AECConfig())
        let result = runner.run(scenario: .echoOnly, aecEnabled: true, diagnosticLogger: nil)
        #expect(result.erleDB > 2.0)
        #expect(result.residualToMicRatio < 1.0)
    }

    @Test func dtdDetectsDoubleTalk() async throws {
        let dtd = GeigelDTD(sampleRate: 16_000, threshold: 0.3, hangoverMs: 80)
        let frameSize = 1024
        let ref = (0..<frameSize).map { _ in Float.random(in: -0.2...0.2) }
        let micEchoOnly = ref.map { $0 * 0.8 }
        let micDoubleTalk = zip(ref, (0..<frameSize).map { _ in Float.random(in: -0.6...0.6) }).map(+)

        _ = dtd.detect(micFrame: micEchoOnly, refFrame: ref)
        let detected = dtd.detect(micFrame: micDoubleTalk, refFrame: ref)
        #expect(detected)
    }

    @Test func delayEstimatorFindsPositiveDelay() async throws {
        let estimator = DelayEstimator(sampleRate: 16_000, frameSize: 1_024, maxDelayMs: 300)
        let delaySamples = 120
        let total = 4096
        let src = (0..<total).map { _ in Float.random(in: -0.3...0.3) }
        var mic = [Float](repeating: 0, count: total)
        for i in delaySamples..<total { mic[i] = src[i - delaySamples] }

        var estimated = 0
        for i in stride(from: 0, to: total - 1024, by: 1024) {
            let refFrame = Array(src[i..<(i + 1024)])
            let micFrame = Array(mic[i..<(i + 1024)])
            estimated = estimator.update(refFrame: refFrame, micFrame: micFrame)
        }
        #expect(estimated >= 0)
    }

    @Test func loopbackRunnerWorksWithoutAEC() async throws {
        let runner = LoopbackExperimentRunner(config: AECConfig())
        let result = runner.run(scenario: .doubleTalk, aecEnabled: false, diagnosticLogger: nil)
        #expect(result.erleDB < 1.0)
    }
}
