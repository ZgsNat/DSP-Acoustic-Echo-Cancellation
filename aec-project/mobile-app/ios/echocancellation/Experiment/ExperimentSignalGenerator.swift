import Foundation

enum ExperimentSignalGenerator {
    static func generateRIR(length: Int = 4096, sampleRate: Int = 16_000, seed: UInt64 = 42) -> [Float] {
        var rir = [Float](repeating: 0, count: length)
        let direct = Int(0.005 * Float(sampleRate))
        if direct < length { rir[direct] = 1.0 }

        let reflections: [(Int, Float)] = [
            (Int(0.012 * Float(sampleRate)), 0.6),
            (Int(0.020 * Float(sampleRate)), 0.4),
            (Int(0.035 * Float(sampleRate)), 0.25),
            (Int(0.050 * Float(sampleRate)), 0.15)
        ]
        for (d, g) in reflections where d < length { rir[d] = g }

        let decayStart = Int(0.060 * Float(sampleRate))
        if decayStart < length {
            var rng = LCG(seed: seed)
            let rt60: Float = 0.2
            for i in decayStart..<length {
                let t = Float(i - decayStart) / Float(sampleRate)
                let decay = exp(-6.9 * t / rt60)
                let noise = (rng.nextFloat() * 2 - 1) * 0.05
                rir[i] = noise * decay
            }
        }

        let peak = max(1e-10, rir.map(abs).max() ?? 1)
        return rir.map { $0 / peak }
    }

    static func generateSpeechLike(
        durationSec: Float,
        sampleRate: Int = 16_000,
        pauseRatio: Float,
        seed: UInt64
    ) -> [Float] {
        let n = Int(durationSec * Float(sampleRate))
        var rng = LCG(seed: seed)

        var white = [Float](repeating: 0, count: n)
        for i in 0..<n { white[i] = rng.nextFloat() * 2 - 1 }

        // Lightweight band shaping (~300-3400Hz at 16k) via cascaded one-pole filters.
        var hp = [Float](repeating: 0, count: n)
        var yPrev: Float = 0
        var xPrev: Float = 0
        let hpAlpha: Float = 0.9
        for i in 0..<n {
            let y = hpAlpha * (yPrev + white[i] - xPrev)
            hp[i] = y
            yPrev = y
            xPrev = white[i]
        }

        var bp = [Float](repeating: 0, count: n)
        let lpAlpha: Float = 0.2
        var lpPrev: Float = 0
        for i in 0..<n {
            lpPrev = lpPrev + lpAlpha * (hp[i] - lpPrev)
            bp[i] = lpPrev
        }

        let fAM = 4 + rng.nextFloat() * 4
        let phase = rng.nextFloat() * 2 * Float.pi
        for i in 0..<n {
            let t = Float(i) / Float(sampleRate)
            let envelope = 0.5 * (1 + sin(2 * Float.pi * fAM * t + phase))
            bp[i] *= envelope
        }

        if pauseRatio > 0 {
            var pos = 0
            var speaking = true
            while pos < n {
                let segMs = Int(rng.nextFloat(in: 200...800))
                let segLen = min(Int(Float(segMs) * Float(sampleRate) / 1000), n - pos)
                if !speaking {
                    for i in pos..<(pos + segLen) { bp[i] = 0 }
                }
                pos += segLen
                if speaking {
                    speaking = rng.nextFloat() > pauseRatio
                } else {
                    speaking = rng.nextFloat() > 0.4
                }
            }
        }

        let peak = max(1e-10, bp.map(abs).max() ?? 1)
        return bp.map { 0.3 * $0 / peak }
    }

    static func convolve(signal: [Float], rir: [Float]) -> [Float] {
        guard !signal.isEmpty, !rir.isEmpty else { return signal }
        var out = [Float](repeating: 0, count: signal.count)
        for i in 0..<signal.count {
            var acc: Float = 0
            let kmax = min(i, rir.count - 1)
            for k in 0...kmax {
                acc += signal[i - k] * rir[k]
            }
            out[i] = acc
        }
        return out
    }
}

struct LCG {
    private var state: UInt64
    init(seed: UInt64) { self.state = seed == 0 ? 1 : seed }
    mutating func next() -> UInt64 {
        state = 2862933555777941757 &* state &+ 3037000493
        return state
    }
    mutating func nextFloat() -> Float {
        Float(next() & 0xFFFFFF) / Float(0xFFFFFF)
    }
    mutating func nextFloat(in range: ClosedRange<Float>) -> Float {
        range.lowerBound + (range.upperBound - range.lowerBound) * nextFloat()
    }
}
