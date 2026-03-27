import Foundation

final class NonlinearSuppressor {
    private let hopSize: Int
    private let windowSize: Int
    private let alpha: Float
    private let beta: Float
    private let smoothAlpha: Float
    private let window: [Float]
    private let dft: DFT

    private var eBuffer: [Float]
    private var yBuffer: [Float]
    private var rBuffer: [Float]
    private var olaBuffer: [Float]

    private var echoPowerSmooth: [Float]?
    private var refPowerSmooth: [Float]?
    private var gainSmooth: [Float]?
    private var echoRefRatio: Float = 0.5

    init(frameSize: Int = 1024, alpha: Float = 2.5, beta: Float = 0.005, smoothAlpha: Float = 0.85) {
        self.hopSize = frameSize
        self.windowSize = frameSize * 2
        self.alpha = alpha
        self.beta = beta
        self.smoothAlpha = smoothAlpha
        self.window = DSPMath.hanning(frameSize * 2)
        self.dft = DFT(length: frameSize * 2)!
        self.eBuffer = [Float](repeating: 0, count: frameSize * 2)
        self.yBuffer = [Float](repeating: 0, count: frameSize * 2)
        self.rBuffer = [Float](repeating: 0, count: frameSize * 2)
        self.olaBuffer = [Float](repeating: 0, count: frameSize * 2)
    }

    func process(
        residualFrame: [Float],
        echoEstimateFrame: [Float],
        refFrame: [Float]?,
        isDoubleTalk: Bool
    ) -> [Float] {
        guard residualFrame.count == hopSize, echoEstimateFrame.count == hopSize else {
            return residualFrame
        }

        shiftAppend(&eBuffer, with: residualFrame)
        shiftAppend(&yBuffer, with: echoEstimateFrame)

        if isDoubleTalk {
            trackEchoPowerFromY()
            if let refFrame { trackRefPower(refFrame) }
            for i in olaBuffer.indices { olaBuffer[i] = 0 }
            gainSmooth = nil
            return residualFrame
        }

        let eWin = zip(eBuffer, window).map { $0 * $1 }
        let yWin = zip(yBuffer, window).map { $0 * $1 }
        let eFFT = dft.forwardReal(eWin)
        let yFFT = dft.forwardReal(yWin)
        let ePower = zip(eFFT.real, eFFT.imag).map { $0 * $0 + $1 * $1 }
        let yPower = zip(yFFT.real, yFFT.imag).map { $0 * $0 + $1 * $1 }

        if echoPowerSmooth == nil {
            echoPowerSmooth = yPower
        } else {
            blend(into: &echoPowerSmooth!, with: yPower, alpha: smoothAlpha)
        }
        var echoEffective = echoPowerSmooth!

        if let refFrame {
            trackRefPower(refFrame)
            let refSmooth = refPowerSmooth ?? [Float](repeating: 0, count: windowSize)
            let echoFromRef = refSmooth.map { echoRefRatio * $0 }
            echoEffective = zip(echoEffective, echoFromRef).map(max)

            let totalY = max(1e-10, yPower.reduce(0, +))
            let totalR = max(1e-10, refSmooth.reduce(0, +))
            let ratio = min(5.0, max(0.01, totalY / totalR))
            echoRefRatio = 0.95 * echoRefRatio + 0.05 * ratio
        }

        let eps: Float = 1e-10
        var gain = [Float](repeating: 1, count: windowSize)
        for i in 0..<windowSize {
            gain[i] = 1 - alpha * echoEffective[i] / (ePower[i] + eps)
            gain[i] = min(1, max(beta, gain[i]))
        }

        if gainSmooth == nil {
            gainSmooth = gain
        } else {
            let attack: Float = 0.1
            let release: Float = 0.85
            for i in 0..<windowSize {
                let prev = gainSmooth![i]
                let a = gain[i] < prev ? attack : release
                gainSmooth![i] = a * prev + (1 - a) * gain[i]
            }
        }

        var freqSmooth = gainSmooth!
        if windowSize > 2 {
            for i in 1..<(windowSize - 1) {
                freqSmooth[i] = 0.25 * gainSmooth![i - 1] + 0.5 * gainSmooth![i] + 0.25 * gainSmooth![i + 1]
            }
            DSPMath.clip(&freqSmooth, min: beta, max: 1)
        }

        let cleanReal = zip(eFFT.real, freqSmooth).map { $0 * $1 }
        let cleanImag = zip(eFFT.imag, freqSmooth).map { $0 * $1 }
        let eCleanWin = dft.inverseComplex(real: cleanReal, imag: cleanImag)

        for i in 0..<windowSize { olaBuffer[i] += eCleanWin[i] }
        let out = Array(olaBuffer[0..<hopSize])
        for i in 0..<hopSize { olaBuffer[i] = olaBuffer[i + hopSize] }
        for i in hopSize..<windowSize { olaBuffer[i] = 0 }
        return out
    }

    func reset() {
        echoPowerSmooth = nil
        refPowerSmooth = nil
        gainSmooth = nil
        echoRefRatio = 0.5
        eBuffer = [Float](repeating: 0, count: windowSize)
        yBuffer = [Float](repeating: 0, count: windowSize)
        rBuffer = [Float](repeating: 0, count: windowSize)
        olaBuffer = [Float](repeating: 0, count: windowSize)
    }

    private func trackEchoPowerFromY() {
        let yWin = zip(yBuffer, window).map { $0 * $1 }
        let yFFT = dft.forwardReal(yWin)
        let yPower = zip(yFFT.real, yFFT.imag).map { $0 * $0 + $1 * $1 }
        if echoPowerSmooth == nil {
            echoPowerSmooth = yPower
        } else {
            blend(into: &echoPowerSmooth!, with: yPower, alpha: smoothAlpha)
        }
    }

    private func trackRefPower(_ refFrame: [Float]) {
        guard refFrame.count == hopSize else { return }
        shiftAppend(&rBuffer, with: refFrame)
        let rWin = zip(rBuffer, window).map { $0 * $1 }
        let rFFT = dft.forwardReal(rWin)
        let rPower = zip(rFFT.real, rFFT.imag).map { $0 * $0 + $1 * $1 }
        if refPowerSmooth == nil {
            refPowerSmooth = rPower
        } else {
            blend(into: &refPowerSmooth!, with: rPower, alpha: smoothAlpha)
        }
    }

    private func shiftAppend(_ buf: inout [Float], with hop: [Float]) {
        for i in 0..<hopSize { buf[i] = buf[i + hopSize] }
        for i in 0..<hopSize { buf[hopSize + i] = hop[i] }
    }

    private func blend(into dst: inout [Float], with src: [Float], alpha: Float) {
        for i in 0..<dst.count { dst[i] = alpha * dst[i] + (1 - alpha) * src[i] }
    }
}
