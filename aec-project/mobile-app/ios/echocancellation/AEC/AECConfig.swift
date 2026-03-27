import Foundation

struct AECConfig {
    var sampleRate: Int = 16_000
    var frameSize: Int = 1_024
    var filterLength: Int = 4_096
    var mu: Float = 0.5
    var eps: Float = 1e-6
    var maxDelayMs: Float = 300
    var dtdThreshold: Float = 0.8
    var dtdHangoverMs: Float = 100
    var nlsAlpha: Float = 2.5
    var nlsBeta: Float = 0.005
}

struct NLMSConfig {
    var filterLength: Int = 512
    var mu: Float = 0.1
    var eps: Float = 1e-6
}
