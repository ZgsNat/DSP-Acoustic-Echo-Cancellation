import Foundation
import Accelerate

enum DSPMath {
    static func rms(_ x: [Float]) -> Float {
        guard !x.isEmpty else { return 0 }
        var sum: Float = 0
        vDSP_svesq(x, 1, &sum, vDSP_Length(x.count))
        return sqrt(sum / Float(x.count))
    }

    static func meanPower(_ x: [Float]) -> Float {
        guard !x.isEmpty else { return 0 }
        var sum: Float = 0
        vDSP_svesq(x, 1, &sum, vDSP_Length(x.count))
        return sum / Float(x.count)
    }

    static func dot(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count)
        var value: Float = 0
        vDSP_dotpr(a, 1, b, 1, &value, vDSP_Length(a.count))
        return value
    }

    static func clip(_ x: inout [Float], min lo: Float, max hi: Float) {
        var low = lo
        var high = hi
        vDSP_vclip(x, 1, &low, &high, &x, 1, vDSP_Length(x.count))
    }

    static func hanning(_ count: Int) -> [Float] {
        guard count > 0 else { return [] }
        var window = [Float](repeating: 0, count: count)
        vDSP_hann_window(&window, vDSP_Length(count), Int32(vDSP_HANN_NORM))
        return window
    }
}

final class DFT {
    private let length: Int
    private let forward: vDSP_DFT_Setup
    private let inverse: vDSP_DFT_Setup

    init?(length: Int) {
        guard length > 0,
              let f = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(length), .FORWARD),
              let i = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(length), .INVERSE) else {
            return nil
        }
        self.length = length
        self.forward = f
        self.inverse = i
    }

    deinit {
        vDSP_DFT_DestroySetup(forward)
        vDSP_DFT_DestroySetup(inverse)
    }

    func forwardReal(_ input: [Float]) -> (real: [Float], imag: [Float]) {
        precondition(input.count == length)
        var inReal = input
        var inImag = [Float](repeating: 0, count: length)
        var outReal = [Float](repeating: 0, count: length)
        var outImag = [Float](repeating: 0, count: length)
        vDSP_DFT_Execute(forward, &inReal, &inImag, &outReal, &outImag)
        return (outReal, outImag)
    }

    func inverseComplex(real: [Float], imag: [Float]) -> [Float] {
        precondition(real.count == length && imag.count == length)
        var inReal = real
        var inImag = imag
        var outReal = [Float](repeating: 0, count: length)
        var outImag = [Float](repeating: 0, count: length)
        vDSP_DFT_Execute(inverse, &inReal, &inImag, &outReal, &outImag)
        var scale = Float(1.0 / Float(length))
        vDSP_vsmul(outReal, 1, &scale, &outReal, 1, vDSP_Length(length))
        return outReal
    }
}
