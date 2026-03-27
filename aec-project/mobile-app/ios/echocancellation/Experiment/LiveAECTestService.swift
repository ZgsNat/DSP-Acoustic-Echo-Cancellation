import Foundation
import AVFAudio

final class LiveAECTestService {
    struct LiveSnapshot {
        let erleDB: Float
        let delayMs: Float
        let dtRatioPercent: Float
        let filterNorm: Float
    }

    var onSnapshot: ((LiveSnapshot) -> Void)?
    var onStatus: ((String) -> Void)?
    var onError: ((String) -> Void)?

    private let cfg = AECConfig(
        sampleRate: 16_000,
        frameSize: 1_024,
        filterLength: 1_024,
        mu: 0.5,
        eps: 1e-6,
        maxDelayMs: 300,
        dtdThreshold: 0.8,
        dtdHangoverMs: 100,
        nlsAlpha: 2.5,
        nlsBeta: 0.005
    )
    private let queue = DispatchQueue(label: "aec.live.service.queue")
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private var sourceTimer: DispatchSourceTimer?
    private var processTimer: DispatchSourceTimer?
    private var converter: AVAudioConverter?
    private var pipeline: AECPipeline?
    private var logger: DiagnosticLogger?
    private var isAECEnabled: Bool = true
    private var running = false

    private var farSignal: [Float] = []
    private var farReadIndex: Int = 0
    private var micAccum: [Float] = []
    private var micQueue: [[Float]] = []
    private var refQueue: [[Float]] = []
    private var lastRef: [Float] = []
    private var processedFrames = 0

    private lazy var targetFormat: AVAudioFormat = {
        AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(cfg.sampleRate), channels: 1, interleaved: false)!
    }()

    func start(aecEnabled: Bool) {
        queue.async {
            if self.running { return }
            self.running = true
            self.isAECEnabled = aecEnabled
            self.processedFrames = 0
            self.micAccum.removeAll(keepingCapacity: true)
            self.micQueue.removeAll(keepingCapacity: true)
            self.refQueue.removeAll(keepingCapacity: true)
            self.lastRef = [Float](repeating: 0, count: self.cfg.frameSize)
            self.farSignal = ExperimentSignalGenerator.generateSpeechLike(
                durationSec: 20,
                sampleRate: self.cfg.sampleRate,
                pauseRatio: 0.2,
                seed: 12345
            )
            self.farReadIndex = 0
            self.logger = DiagnosticLogger(fileName: "aec_live_\(Int(Date().timeIntervalSince1970)).csv")
            self.pipeline = AECPipeline(config: self.cfg, diagnosticLogger: self.logger)
            self.setupAudioSessionAndEngine()
        }
    }

    func updateAECEnabled(_ value: Bool) {
        queue.async {
            self.isAECEnabled = value
            if value {
                self.pipeline?.reset()
            }
        }
    }

    func stop() {
        queue.async {
            guard self.running else { return }
            self.running = false
            self.sourceTimer?.cancel()
            self.processTimer?.cancel()
            self.sourceTimer = nil
            self.processTimer = nil
            self.engine.inputNode.removeTap(onBus: 0)
            self.player.stop()
            self.engine.stop()
            let summary = self.logger?.summary() ?? "No live summary."
            let path = self.logger?.exportURL().path ?? ""
            self.onStatus?("Live stopped. \(summary)")
            if !path.isEmpty {
                self.onStatus?("Live CSV: \(path)")
            }
        }
    }

    private func setupAudioSessionAndEngine() {
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playAndRecord, mode: .voiceChat, options: [.defaultToSpeaker, .allowBluetooth])
            try session.setPreferredSampleRate(Double(cfg.sampleRate))
            try session.setActive(true)
        } catch {
            running = false
            onError?("AudioSession error: \(error.localizedDescription)")
            return
        }

        session.requestRecordPermission { [weak self] granted in
            guard let self else { return }
            self.queue.async {
                guard self.running else { return }
                guard granted else {
                    self.running = false
                    self.onError?("Microphone permission denied.")
                    return
                }
                self.startEngineGraph()
            }
        }
    }

    private func startEngineGraph() {
        engine.stop()
        engine.reset()

        let inputNode = engine.inputNode
        let inputFormat = inputNode.inputFormat(forBus: 0)
        converter = AVAudioConverter(from: inputFormat, to: targetFormat)

        if player.engine == nil {
            engine.attach(player)
            engine.connect(player, to: engine.mainMixerNode, format: targetFormat)
        }

        inputNode.removeTap(onBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 2048, format: inputFormat) { [weak self] buffer, _ in
            self?.handleInputBuffer(buffer)
        }

        do {
            try engine.start()
            player.play()
            startTimers()
            onStatus?("Live started. Speak near the phone while reference plays from speaker.")
        } catch {
            running = false
            onError?("Engine start failed: \(error.localizedDescription)")
        }
    }

    private func startTimers() {
        let frameInterval = Double(cfg.frameSize) / Double(cfg.sampleRate)

        let src = DispatchSource.makeTimerSource(queue: queue)
        src.schedule(deadline: .now(), repeating: frameInterval)
        src.setEventHandler { [weak self] in
            self?.produceReferenceFrame()
        }
        src.resume()
        sourceTimer = src

        let proc = DispatchSource.makeTimerSource(queue: queue)
        proc.schedule(deadline: .now() + frameInterval / 2, repeating: frameInterval)
        proc.setEventHandler { [weak self] in
            self?.processOneFrame()
        }
        proc.resume()
        processTimer = proc
    }

    private func produceReferenceFrame() {
        guard running else { return }
        let frame = nextReferenceFrame()
        refQueue.append(frame)
        if refQueue.count > 16 { refQueue.removeFirst(refQueue.count - 16) }
        schedulePlayback(frame: frame)
    }

    private func processOneFrame() {
        guard running else { return }
        guard let pipeline else { return }

        let mic = micQueue.isEmpty ? [Float](repeating: 0, count: cfg.frameSize) : micQueue.removeFirst()
        let refFrames = refQueue
        refQueue.removeAll(keepingCapacity: true)
        let refFrame: [Float]
        let refIsSilence: Bool

        if refFrames.isEmpty {
            refFrame = lastRef
            refIsSilence = (refFrame.map(abs).max() ?? 0) < 1e-6
        } else {
            if isAECEnabled && refFrames.count > 1 {
                for mid in refFrames.dropLast() {
                    pipeline.feedReference(mid)
                }
            }
            refFrame = refFrames.last ?? lastRef
            lastRef = refFrame
            refIsSilence = false
        }

        pipeline.markRefSilence(refIsSilence)
        _ = isAECEnabled ? pipeline.process(micFrame: mic, refFrame: refFrame) : mic
        processedFrames += 1

        if processedFrames % 8 == 0 {
            let m = pipeline.getMetrics(reset: false)
            onSnapshot?(
                LiveSnapshot(
                    erleDB: m.erleDB,
                    delayMs: m.delayMs,
                    dtRatioPercent: m.doubleTalkRatio * 100,
                    filterNorm: m.filterNorm
                )
            )
        }
    }

    private func handleInputBuffer(_ buffer: AVAudioPCMBuffer) {
        queue.async {
            guard self.running else { return }
            guard let converted = self.convertToTarget(buffer) else { return }
            guard let ch = converted.floatChannelData?.pointee else { return }
            let count = Int(converted.frameLength)
            if count <= 0 { return }
            self.micAccum.append(contentsOf: UnsafeBufferPointer(start: ch, count: count))
            while self.micAccum.count >= self.cfg.frameSize {
                let frame = Array(self.micAccum.prefix(self.cfg.frameSize))
                self.micAccum.removeFirst(self.cfg.frameSize)
                self.micQueue.append(frame)
                if self.micQueue.count > 16 {
                    self.micQueue.removeFirst(self.micQueue.count - 16)
                }
            }
        }
    }

    private func convertToTarget(_ input: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
        guard let converter else { return nil }
        let inFrames = input.frameLength
        let ratio = targetFormat.sampleRate / input.format.sampleRate
        let outCapacity = AVAudioFrameCount(Double(inFrames) * ratio + 16)
        guard let out = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outCapacity) else { return nil }

        var consumed = false
        let status = converter.convert(to: out, error: nil) { _, outStatus in
            if consumed {
                outStatus.pointee = .noDataNow
                return nil
            }
            consumed = true
            outStatus.pointee = .haveData
            return input
        }
        if status == .haveData || status == .inputRanDry {
            return out
        }
        return nil
    }

    private func nextReferenceFrame() -> [Float] {
        if farSignal.isEmpty {
            return [Float](repeating: 0, count: cfg.frameSize)
        }
        var out = [Float](repeating: 0, count: cfg.frameSize)
        for i in 0..<cfg.frameSize {
            out[i] = farSignal[farReadIndex]
            farReadIndex += 1
            if farReadIndex >= farSignal.count { farReadIndex = 0 }
        }
        return out
    }

    private func schedulePlayback(frame: [Float]) {
        guard let pcm = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: AVAudioFrameCount(frame.count)) else { return }
        pcm.frameLength = AVAudioFrameCount(frame.count)
        if let ch = pcm.floatChannelData?.pointee {
            frame.withUnsafeBufferPointer { src in
                ch.update(from: src.baseAddress!, count: frame.count)
            }
            player.scheduleBuffer(pcm, completionHandler: nil)
        }
    }
}
