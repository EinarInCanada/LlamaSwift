//
//  MTMDContext.swift
//  Gemholic
//
//  Swift bridge over libmtmd (llama.cpp multimodal). Owns the mtmd_context
//  lifecycle and wraps the C primitives the app needs: tokenize prompt +
//  bitmaps into chunks, encode a chunk, read back embeddings, introspect
//  chunk metadata for the decode loop.
//
//  Orchestration (prompt → mtmd_encode → llama_decode) lives in LlamaContext;
//  this file intentionally stays a thin, Swift-idiomatic wrapper.
//

import Foundation
import UIKit

enum MTMDError: Error {
    case initFailed
    case markerMismatch
    case preprocessFailed
    case encodeFailed
    case unknownTokenize(Int32)
}

enum MTMDChunkType {
    case text, image, audio

    init?(raw: mtmd_input_chunk_type) {
        switch raw {
        case MTMD_INPUT_CHUNK_TYPE_TEXT:  self = .text
        case MTMD_INPUT_CHUNK_TYPE_IMAGE: self = .image
        case MTMD_INPUT_CHUNK_TYPE_AUDIO: self = .audio
        default: return nil
        }
    }
}

final class MTMDBitmap {
    fileprivate let ptr: OpaquePointer

    /// RGB image bitmap. `rgb.count` must equal width * height * 3.
    init(rgb: Data, width: UInt32, height: UInt32) {
        self.ptr = rgb.withUnsafeBytes { raw -> OpaquePointer in
            let base = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
            return mtmd_bitmap_init(width, height, base)
        }
    }

    /// PCM F32 mono audio, typically 16 kHz for Whisper-style encoders.
    init(pcmF32: [Float]) {
        self.ptr = pcmF32.withUnsafeBufferPointer { buf in
            mtmd_bitmap_init_from_audio(buf.count, buf.baseAddress)
        }
    }

    var isAudio: Bool { mtmd_bitmap_is_audio(ptr) }

    func setId(_ id: String) {
        id.withCString { mtmd_bitmap_set_id(ptr, $0) }
    }

    deinit { mtmd_bitmap_free(ptr) }

    /// Decode `imageData` (e.g. JPEG/PNG/HEIC bytes) into a packed RGB24 buffer
    /// and wrap it. Clamps the largest side to `maxDim` to cap memory before
    /// mtmd preprocesses further. Returns nil if decoding fails.
    static func fromImageData(_ imageData: Data, maxDim: Int = 1024) -> MTMDBitmap? {
        guard let uiImage = UIImage(data: imageData), let cgImage = uiImage.cgImage else {
            return nil
        }

        var w = cgImage.width
        var h = cgImage.height
        let longest = max(w, h)
        if longest > maxDim {
            let scale = Double(maxDim) / Double(longest)
            w = max(1, Int((Double(w) * scale).rounded()))
            h = max(1, Int((Double(h) * scale).rounded()))
        }

        // CGBitmapContext has no 24bpp RGB format; draw into 32bpp RGBA then pack.
        let rgbaStride = w * 4
        var rgba = Data(count: rgbaStride * h)
        let drew: Bool = rgba.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress else { return false }
            let cs = CGColorSpaceCreateDeviceRGB()
            let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
            guard let ctx = CGContext(
                data: base, width: w, height: h,
                bitsPerComponent: 8, bytesPerRow: rgbaStride,
                space: cs, bitmapInfo: bitmapInfo
            ) else { return false }
            ctx.interpolationQuality = .medium
            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard drew else { return nil }

        let rgbStride = w * 3
        var rgb = Data(count: rgbStride * h)
        rgb.withUnsafeMutableBytes { outRaw in
            rgba.withUnsafeBytes { inRaw in
                let dst = outRaw.baseAddress!.assumingMemoryBound(to: UInt8.self)
                let src = inRaw.baseAddress!.assumingMemoryBound(to: UInt8.self)
                let pixels = w * h
                for i in 0..<pixels {
                    dst[i * 3 + 0] = src[i * 4 + 0]
                    dst[i * 3 + 1] = src[i * 4 + 1]
                    dst[i * 3 + 2] = src[i * 4 + 2]
                }
            }
        }
        return MTMDBitmap(rgb: rgb, width: UInt32(w), height: UInt32(h))
    }
}

/// Opaque chunk returned by tokenize. Lifetime is tied to its parent MTMDChunks.
struct MTMDChunk {
    fileprivate let ptr: OpaquePointer

    var type: MTMDChunkType {
        MTMDChunkType(raw: mtmd_input_chunk_get_type(ptr)) ?? .text
    }

    /// Number of embedding slots this chunk will occupy in the llama KV cache.
    var nTokens: Int { mtmd_input_chunk_get_n_tokens(ptr) }

    /// Number of temporal positions to advance n_past by (differs from
    /// nTokens only for M-RoPE image models).
    var nPos: llama_pos { mtmd_input_chunk_get_n_pos(ptr) }

    /// Only valid when type == .text.
    var textTokens: [llama_token] {
        var count: size_t = 0
        guard let base = mtmd_input_chunk_get_tokens_text(ptr, &count), count > 0 else {
            return []
        }
        return Array(UnsafeBufferPointer(start: base, count: count))
    }
}

final class MTMDChunks {
    fileprivate let ptr: OpaquePointer

    fileprivate init(_ ptr: OpaquePointer) { self.ptr = ptr }

    var count: Int { mtmd_input_chunks_size(ptr) }

    subscript(index: Int) -> MTMDChunk {
        let p = mtmd_input_chunks_get(ptr, index)!
        return MTMDChunk(ptr: p)
    }

    deinit { mtmd_input_chunks_free(ptr) }
}

/// Owns the mtmd_context. Must be held alive for as long as any embeddings
/// retrieved via `outputEmbedPointer()` are in use — libmtmd reuses the
/// internal buffer on the next encode.
final class MTMDContext {
    private let ptr: OpaquePointer

    /// - Parameters:
    ///   - mmprojPath: path to the *.mmproj.gguf companion file
    ///   - textModel: the already-loaded llama_model pointer (from LlamaContext)
    ///   - useGPU: Metal backend for the vision/audio tower (iOS: true)
    ///   - threads: CPU threads for preprocessing/fallback paths
    init(mmprojPath: String, textModel: OpaquePointer, useGPU: Bool = true, threads: Int32 = 4) throws {
        var params = mtmd_context_params_default()
        params.use_gpu = useGPU
        params.print_timings = false
        params.n_threads = threads
        params.warmup = false

        guard let ctx = mtmd_init_from_file(mmprojPath, textModel, params) else {
            throw MTMDError.initFailed
        }
        self.ptr = ctx
    }

    var supportsVision: Bool { mtmd_support_vision(ptr) }
    var supportsAudio: Bool  { mtmd_support_audio(ptr) }
    var useNonCausalMask: Bool { mtmd_decode_use_non_causal(ptr, nil) }
    var useMRoPE: Bool { mtmd_decode_use_mrope(ptr) }
    var audioSampleRate: Int32 { mtmd_get_audio_sample_rate(ptr) }

    /// Marker string to embed in the prompt where bitmaps should be spliced in.
    /// Default: "<__media__>".
    static var mediaMarker: String { String(cString: mtmd_default_marker()) }

    /// Tokenize `prompt` against `bitmaps`. The marker count in prompt must
    /// equal bitmaps.count.
    func tokenize(prompt: String, bitmaps: [MTMDBitmap], addSpecial: Bool = true) throws -> MTMDChunks {
        let outChunks = mtmd_input_chunks_init()!

        var bitmapPtrs: [OpaquePointer?] = bitmaps.map { Optional($0.ptr) }

        let status: Int32 = prompt.withCString { cPrompt in
            var input = mtmd_input_text(
                text: cPrompt,
                add_special: addSpecial,
                parse_special: true
            )
            return bitmapPtrs.withUnsafeMutableBufferPointer { buf in
                mtmd_tokenize(ptr, outChunks, &input, buf.baseAddress, bitmaps.count)
            }
        }

        guard status == 0 else {
            mtmd_input_chunks_free(outChunks)
            switch status {
            case 1: throw MTMDError.markerMismatch
            case 2: throw MTMDError.preprocessFailed
            default: throw MTMDError.unknownTokenize(status)
            }
        }
        return MTMDChunks(outChunks)
    }

    /// Encode a single image or audio chunk. After this returns, call
    /// `outputEmbedPointer()` immediately — the buffer is overwritten on the
    /// next encode.
    func encode(chunk: MTMDChunk) throws {
        let rc = mtmd_encode_chunk(ptr, chunk.ptr)
        guard rc == 0 else { throw MTMDError.encodeFailed }
    }

    /// Pointer to the last encode's output embeddings. Valid size (in floats):
    /// `llama_model_n_embd_inp(model) * chunk.nTokens`. Caller must feed these
    /// into a llama_batch with `embd` set before the next encode call.
    func outputEmbedPointer() -> UnsafeMutablePointer<Float>? {
        mtmd_get_output_embd(ptr)
    }

    deinit { mtmd_free(ptr) }
}
