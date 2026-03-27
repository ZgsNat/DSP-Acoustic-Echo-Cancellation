import Foundation

final class GeigelDTD {
    private let sampleRate: Int
    private let threshold: Float
    private let hangoverMs: Float
    private let echoTailMs: Float

    private var micBuf: [Float] = []
    private var refBuf: [Float] = []
    private var refBufMaxLen: Int = 8
    private var refConfigured = false

    private var hangoverFramesLeft = 0
    private var inDoubleTalk = false

    private var echoGainEstimate: Float = 1.0
    private let echoGainAlpha: Float = 0.95

    init(
        sampleRate: Int = 16_000,
        threshold: Float = 0.5,
        hangoverMs: Float = 100,
        echoTailMs: Float = 300
    ) {
        self.sampleRate = sampleRate
        self.threshold = threshold
        self.hangoverMs = hangoverMs
        self.echoTailMs = echoTailMs
    }

    func detect(micFrame: [Float], refFrame: [Float]) -> Bool {
        if !refConfigured {
            let frameMs = Float(micFrame.count) / Float(sampleRate) * 1000
            refBufMaxLen = max(Int(ceil(echoTailMs / frameMs)), 2)
            refConfigured = true
        }

        let micRMS = DSPMath.rms(micFrame)
        let refRMS = DSPMath.rms(refFrame)
        append(&micBuf, value: micRMS, maxLen: 2)
        append(&refBuf, value: refRMS, maxLen: refBufMaxLen)

        let maxMic = micBuf.max() ?? 0
        let maxRef = refBuf.max() ?? 0
        let minLevel: Float = 1e-3
        let rawDT: Bool

        if maxMic < minLevel && maxRef < minLevel {
            rawDT = false
        } else if maxRef < minLevel {
            rawDT = false
        } else {
            let expectedEcho = echoGainEstimate * maxRef
            rawDT = maxMic > (1 + threshold) * expectedEcho
            if !rawDT, maxRef > minLevel, maxMic > minLevel {
                let gain = maxMic / maxRef
                if gain > 0.1, gain < 10 {
                    echoGainEstimate = echoGainAlpha * echoGainEstimate + (1 - echoGainAlpha) * gain
                }
            }
        }

        let frameMs = Float(micFrame.count) / Float(sampleRate) * 1000
        let hangoverFramesMax = Int(ceil(hangoverMs / frameMs))
        if rawDT {
            hangoverFramesLeft = hangoverFramesMax
            inDoubleTalk = true
        } else if hangoverFramesLeft > 0 {
            hangoverFramesLeft -= 1
            inDoubleTalk = true
        } else {
            inDoubleTalk = false
        }

        return inDoubleTalk
    }

    var isDoubleTalk: Bool { inDoubleTalk }

    func reset() {
        micBuf.removeAll(keepingCapacity: true)
        refBuf.removeAll(keepingCapacity: true)
        hangoverFramesLeft = 0
        inDoubleTalk = false
        echoGainEstimate = 1.0
        refConfigured = false
    }

    private func append(_ buf: inout [Float], value: Float, maxLen: Int) {
        buf.append(value)
        if buf.count > maxLen { buf.removeFirst(buf.count - maxLen) }
    }
}
