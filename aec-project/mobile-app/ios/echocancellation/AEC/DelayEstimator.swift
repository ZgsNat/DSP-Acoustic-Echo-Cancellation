import Foundation

final class DelayEstimator {
    private let sampleRate: Int
    private let maxDelaySamples: Int
    private let alpha: Float
    private let accFrames: Int
    private let minConfidence: Float
    private let confirmCount: Int
    private let nFFT: Int
    private let dft: DFT

    private var refAcc: [[Float]] = []
    private var micAcc: [[Float]] = []
    private var rPhatSmooth: (real: [Float], imag: [Float])?
    private var currentDelay: Int = 0

    private var candidateDelay: Int = 0
    private var candidateCount: Int = 0
    private var locked: Bool = false

    init(
        sampleRate: Int = 16_000,
        frameSize: Int = 1_024,
        maxDelayMs: Float = 250,
        smoothAlpha: Float = 0.9,
        accFrames: Int = 4,
        minConfidence: Float = 0.15,
        confirmCount: Int = 3
    ) {
        self.sampleRate = sampleRate
        self.maxDelaySamples = Int(maxDelayMs * Float(sampleRate) / 1000)
        self.alpha = smoothAlpha
        self.accFrames = accFrames
        self.minConfidence = minConfidence
        self.confirmCount = confirmCount
        self.nFFT = 2 * frameSize * accFrames
        self.dft = DFT(length: self.nFFT)!
    }

    func update(refFrame: [Float], micFrame: [Float]) -> Int {
        refAcc.append(refFrame)
        micAcc.append(micFrame)

        if refAcc.count >= accFrames {
            let x = refAcc.flatMap { $0 }
            let d = micAcc.flatMap { $0 }
            refAcc.removeAll(keepingCapacity: true)
            micAcc.removeAll(keepingCapacity: true)

            let refEnergy = DSPMath.meanPower(x)
            let micEnergy = DSPMath.meanPower(d)
            if refEnergy < 1e-7 || micEnergy < 1e-7 {
                return currentDelay
            }

            var xPad = x
            xPad.append(contentsOf: [Float](repeating: 0, count: max(0, nFFT - x.count)))
            var dPad = d
            dPad.append(contentsOf: [Float](repeating: 0, count: max(0, nFFT - d.count)))

            let xF = dft.forwardReal(xPad)
            let dF = dft.forwardReal(dPad)

            var rReal = [Float](repeating: 0, count: nFFT)
            var rImag = [Float](repeating: 0, count: nFFT)
            for i in 0..<nFFT {
                let xr = xF.real[i]
                let xi = xF.imag[i]
                let dr = dF.real[i]
                let di = dF.imag[i]
                // X * conj(D)
                let real = xr * dr + xi * di
                let imag = xi * dr - xr * di
                let mag = max(1e-10, hypot(real, imag))
                rReal[i] = real / mag
                rImag[i] = imag / mag
            }

            if var smooth = rPhatSmooth {
                for i in 0..<nFFT {
                    smooth.real[i] = alpha * smooth.real[i] + (1 - alpha) * rReal[i]
                    smooth.imag[i] = alpha * smooth.imag[i] + (1 - alpha) * rImag[i]
                }
                rPhatSmooth = smooth
            } else {
                rPhatSmooth = (rReal, rImag)
            }

            let smooth = rPhatSmooth!
            let gcc = dft.inverseComplex(real: smooth.real, imag: smooth.imag)
            let searchEnd = min(maxDelaySamples, gcc.count - 1)
            if searchEnd <= 0 { return currentDelay }
            let search = Array(gcc[0...searchEnd])
            guard let peak = search.enumerated().max(by: { $0.element < $1.element }) else {
                return currentDelay
            }
            let rawDelay = peak.offset
            let peakVal = peak.element
            let meanVal = max(1e-10, search.map { abs($0) }.reduce(0, +) / Float(search.count))
            let confidence = peakVal / meanVal

            if confidence < minConfidence {
                return currentDelay
            }

            if !locked {
                if abs(rawDelay - candidateDelay) <= 2 {
                    candidateCount += 1
                } else {
                    candidateDelay = rawDelay
                    candidateCount = 1
                }
                if candidateCount >= confirmCount {
                    currentDelay = candidateDelay
                    locked = true
                }
            } else {
                let atLocked = search[min(currentDelay, search.count - 1)]
                let lockStillGood = atLocked > 0.5 * peakVal
                if !lockStillGood {
                    if abs(rawDelay - candidateDelay) <= 2 {
                        candidateCount += 1
                    } else {
                        candidateDelay = rawDelay
                        candidateCount = 1
                    }
                    if candidateCount >= confirmCount {
                        currentDelay = candidateDelay
                        candidateCount = 0
                    }
                }
            }
        }

        return currentDelay
    }

    var currentDelaySamples: Int { currentDelay }
    var currentDelayMs: Float { Float(currentDelay) * 1000 / Float(sampleRate) }
}

final class DelayLine {
    private let maxDelay: Int
    private var buffer: [Float]
    private var writeIndex: Int = 0

    init(maxDelaySamples: Int = 48_000) {
        self.maxDelay = maxDelaySamples
        self.buffer = [Float](repeating: 0, count: maxDelaySamples)
    }

    func process(_ frame: [Float], delay: Int) -> [Float] {
        let n = frame.count
        let d = max(0, min(maxDelay - 1, delay))
        var out = [Float](repeating: 0, count: n)
        for i in 0..<n {
            buffer[writeIndex] = frame[i]
            let readIndex = (writeIndex - d + maxDelay) % maxDelay
            out[i] = buffer[readIndex]
            writeIndex = (writeIndex + 1) % maxDelay
        }
        return out
    }

    func reset() {
        buffer = [Float](repeating: 0, count: maxDelay)
        writeIndex = 0
    }
}
