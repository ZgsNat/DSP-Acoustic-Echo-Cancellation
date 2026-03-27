import Foundation

enum LoopbackScenario: String, CaseIterable, Identifiable {
    case echoOnly = "echo_only"
    case intermittent = "intermittent"
    case doubleTalk = "double_talk"
    case refSilence = "ref_silence"
    case worstCase = "worst_case"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .echoOnly: return "Echo Only"
        case .intermittent: return "Intermittent"
        case .doubleTalk: return "Double Talk"
        case .refSilence: return "Ref Silence"
        case .worstCase: return "Worst Case"
        }
    }
}

struct ScenarioConfig {
    let scenario: LoopbackScenario
    let durationSec: Float
    let farPauseRatio: Float
    let nearPauseRatio: Float
    let nearEnabled: Bool
    let noiseLevel: Float
    let refDropRate: Float
}

struct ScenarioResult: Identifiable {
    let id = UUID()
    let scenario: LoopbackScenario
    let erleDB: Float
    let dtRatio: Float
    let residualToMicRatio: Float
    let maxFilterNorm: Float
    let finalFilterNorm: Float
    let avgProcessMs: Float
    let passed: Bool
    let note: String
}
