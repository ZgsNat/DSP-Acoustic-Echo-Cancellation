import Foundation

final class NLMSFilter {
    private let cfg: NLMSConfig
    private var w: [Float]
    private var history: [Float]
    private var divergeCount = 0
    private let divergeMax = 5

    init(config: NLMSConfig = NLMSConfig()) {
        self.cfg = config
        self.w = [Float](repeating: 0, count: config.filterLength)
        self.history = [Float](repeating: 0, count: max(0, config.filterLength - 1))
    }

    func process(micFrame: [Float], refFrame: [Float], update: Bool = true) -> [Float] {
        let n = micFrame.count
        let l = cfg.filterLength
        guard n > 0, refFrame.count == n, l > 1 else { return micFrame }

        let fullRef = history + refFrame
        if l > 1 {
            history = Array(fullRef.suffix(l - 1))
        }

        var eFrame = [Float](repeating: 0, count: n)

        for i in 0..<n {
            // Sliding window reversed: newest sample first.
            let start = i
            let end = i + l
            let xVec = Array(fullRef[start..<end].reversed())
            let y = DSPMath.dot(w, xVec)
            let e = micFrame[i] - y
            eFrame[i] = e

            if update {
                let norm = DSPMath.dot(xVec, xVec) + cfg.eps
                if norm > 100 * cfg.eps {
                    let gain = cfg.mu * e / norm
                    for k in 0..<l {
                        w[k] += gain * xVec[k]
                    }
                }
            }
        }

        let micEnergy = max(1e-10, micFrame.reduce(0) { $0 + $1 * $1 })
        let resEnergy = eFrame.reduce(0) { $0 + $1 * $1 }

        if micEnergy > 1e-8 && resEnergy > 1.5 * micEnergy {
            divergeCount += 1
            if divergeCount >= divergeMax {
                w = [Float](repeating: 0, count: l)
                divergeCount = 0
                eFrame = micFrame
            }
        } else {
            divergeCount = max(0, divergeCount - 1)
        }

        return eFrame
    }

    func feedReference(_ refFrame: [Float]) {
        let l = cfg.filterLength
        guard l > 1 else { return }
        let fullRef = history + refFrame
        history = Array(fullRef.suffix(l - 1))
    }

    var weightNorm: Float {
        sqrt(max(0, w.reduce(0) { $0 + $1 * $1 }))
    }

    func reset() {
        for i in w.indices { w[i] = 0 }
        for i in history.indices { history[i] = 0 }
        divergeCount = 0
    }
}
