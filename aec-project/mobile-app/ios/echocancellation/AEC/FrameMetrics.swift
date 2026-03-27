import Foundation

struct FrameMetrics {
    var frameIndex: Int = 0
    var timestamp: TimeInterval = 0
    var micRMS: Float = 0
    var micPeak: Float = 0
    var refRMS: Float = 0
    var refPeak: Float = 0
    var refIsSilence: Bool = false
    var delaySamples: Int = 0
    var delayMs: Float = 0
    var isDoubleTalk: Bool = false
    var nlmsResidualRMS: Float = 0
    var echoEstimateRMS: Float = 0
    var nlsOutputRMS: Float = 0
    var nlsBypassed: Bool = false
    var erleInstantDB: Float = 0
    var filterNorm: Float = 0
    var processTimeMs: Float = 0
}

struct AECMetricsSnapshot {
    var erleDB: Float
    var doubleTalkRatio: Float
    var frameCount: Int
    var delayMs: Float
    var filterNorm: Float
}
