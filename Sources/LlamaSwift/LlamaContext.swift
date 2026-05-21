//
//  LlamaContext.swift
//
import Foundation

// ⛔️ 硬终止开关。UI 层调 cancel() 之后,采样/预处理循环每轮检查 isCancelled,
//     一旦为 true 立刻 break,保证模型不再占用 GPU/CPU 持续推理。
//     带锁是因为 UI 主线程与 Task.detached 的生成线程会同时访问。
final class LlamaCancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var _cancelled = false
    var isCancelled: Bool { lock.withLock { _cancelled } }
    func cancel() { lock.withLock { _cancelled = true } }
    func reset() { lock.withLock { _cancelled = false } }
}

final class LlamaContext: @unchecked Sendable {
    var model: OpaquePointer?
    var ctx: OpaquePointer?
    var batch: llama_batch

    // 👁️ mtmd 多模态上下文。只有加载了 mmproj 文件后才非空,
    //     未加载时图片/音频路径会跳过,文字路径完全不受影响。
    var mtmd: MTMDContext?

    var n_past: Int32 = 0
    // 🚀 P1-2b 上下文窗口:模型支持 128K (max_position_embeddings=131072),
    //    我们之前只用 8192 (6%),装不下多图历史。32K 是 8GB+ 设备的舒适区
    //    (KV cache ≈ 1.1GB),为 P1-2c 多图记忆 (10 张图 × 280 token = 2800) 腾足空间。
    let n_ctx: Int32 = 32768
    private let safeBatchSize: Int32 = 1024

    // ⛔️ P0-1 硬终止:每次 generate* 入口 reset 一次,采样/解码循环持续轮询。
    let cancelFlag = LlamaCancelFlag()

    /// UI 层暂停按钮的唯一入口 — 让后台推理循环在下一次 yield 时立刻退出,
    /// 释放 GPU/CPU 资源,避免持续高负荷发热。
    func cancelGeneration() {
        cancelFlag.cancel()
    }

    /// P0-24 采样参数覆盖。nil = 用 sampleLoop 默认 (Gemma 4 官方推荐 1.0/64/0.95)。
    /// ASR 等"任务式"调用应该走低温接近 greedy 的解码 — 在 generate* 入口前 set,
    /// sampleLoop 完成后 clear。
    struct SamplingOverride {
        let temperature: Float
        let topK: Int
        let topP: Float
    }
    var samplingOverride: SamplingOverride? = nil
    
    // ✅ 修复：必须保留初始化方法
    private init(model: OpaquePointer, ctx: OpaquePointer, batch: llama_batch) {
        self.model = model
        self.ctx = ctx
        self.batch = batch
    }
    
    // ✅ 修复：必须保留静态创建方法
    static func create_context(path: String) throws -> LlamaContext {
        llama_backend_init()
        
        var model_params = llama_model_default_params()
        model_params.n_gpu_layers = 99 // 启用 Metal GPU
        
        guard let model = llama_model_load_from_file(path, model_params) else {
            throw NSError(domain: "Llama", code: 1, userInfo: [NSLocalizedDescriptionKey: "无法加载模型文件"])
        }
        
        var ctx_params = llama_context_default_params()
        // P1-2b:用 LlamaContext 上声明的 n_ctx (32K),不再硬编码 8192
        ctx_params.n_ctx = 32768
        ctx_params.n_batch = 1024
        // 🚀 性能优化 1：线程全开！对于现代 iPhone/Mac，4 个线程吞吐量最佳
        ctx_params.n_threads = 4
        ctx_params.n_threads_batch = 4
        
        guard let ctx = llama_init_from_model(model, ctx_params) else {
            llama_model_free(model)
            throw NSError(domain: "Llama", code: 2, userInfo: [NSLocalizedDescriptionKey: "初始化 Context 失败"])
        }
        
        let batch = llama_batch_init(1024, 0, 1)
        return LlamaContext(model: model, ctx: ctx, batch: batch)
    }
    
    // MARK: - 👁️ 视觉/音频多模态支持 (libmtmd)

    /// Load the mmproj companion file (vision/audio tower). Idempotent.
    /// Returns false if the file isn't present or mtmd init fails — caller
    /// should fall back to the text-only path.
    func loadMMProj(path: String) -> Bool {
        guard mtmd == nil else { return true }
        guard let model = model else { return false }
        do {
            mtmd = try MTMDContext(mmprojPath: path, textModel: model)
            print("✅ mmproj 已加载: vision=\(mtmd?.supportsVision ?? false) audio=\(mtmd?.supportsAudio ?? false)")
            return true
        } catch {
            print("⚠️ mmproj 加载失败: \(error)")
            return false
        }
    }

    func unloadMMProj() {
        mtmd = nil
    }

    // MARK: - 文本生成核心逻辑
    
    func generateIncrementalResponse(newText: String, onToken: @escaping (String) -> Bool) async throws {
        guard let model = model, let ctx = ctx else { return }

        // ⛔️ 进入新一轮生成前清掉上一轮遗留的取消信号
        cancelFlag.reset()

        let tokens = tokenize(text: newText, add_bos: (n_past == 0))
        let new_tokens_count = Int32(tokens.count)

        if n_past + new_tokens_count + Int32(1024) > n_ctx {
            print("⚠️ 剩余上下文空间不足 1024，正在重置以获取完整回答空间...")
            self.reset()
            try await generateIncrementalResponse(newText: newText, onToken: onToken)
            return
        }

        self.batch.n_tokens = 0
        for (i, token) in tokens.enumerated() {
            let isLast = (i == tokens.count - 1)
            batch_add(token: token, pos: n_past + Int32(i), logits: isLast)

            if self.batch.n_tokens >= self.safeBatchSize && !isLast {
                // 长 prompt 预处理中途也要响应取消 — 避免用户点暂停后还要等几秒才断
                if cancelFlag.isCancelled {
                    print("⛔️ 预处理中途收到取消信号,停止喂入 prompt")
                    return
                }
                if llama_decode(ctx, batch) != 0 {
                    print("❌ 预处理长 Prompt 失败")
                    return
                }
                self.batch.n_tokens = 0
                await Task.yield()
            }
        }
        n_past += new_tokens_count

        try await sampleLoop(model: model, ctx: ctx, onToken: onToken)
    }

    // MARK: - 🖼️ 带图片 prompt 的流式生成 (libmtmd 路径)
    //
    // 用法:
    //   - newText 中必须包含 MTMDContext.mediaMarker ("<__media__>") 标记,
    //     每个标记对应 images 里的一张图。GemmaEngine 会在 user turn 内注入。
    //   - mtmd 必须已经 loadMMProj 成功。未加载时抛错,上层回退到文字路径。
    //
    // 实现 idea:
    //   prompt + bitmaps --mtmd_tokenize--> [text/image chunk, ...]
    //     对每个 chunk:
    //       text chunk:  逐 token llama_decode, 复用已有 batch
    //       image chunk: mtmd_encode_chunk -> 读 embed 指针 -> 构造 embd batch -> llama_decode
    //     chunk 走完后 n_past 已经指向 user prompt 末尾,和文字路径接轨,
    //     再进入共享 sampleLoop 出字即可。
    func generateIncrementalResponseWithImages(
        newText: String,
        images: [Data],
        onToken: @escaping (String) -> Bool
    ) async throws {
        let bitmaps: [MTMDBitmap] = images.compactMap { MTMDBitmap.fromImageData($0) }
        guard bitmaps.count == images.count else {
            throw NSError(domain: "LlamaContext", code: 11, userInfo: [NSLocalizedDescriptionKey: "图片解码失败"])
        }
        try await generateIncrementalResponseWithMedia(newText: newText, bitmaps: bitmaps, onToken: onToken)
    }

    /// 音频路径:samples 是每段录音 16 kHz 单声道 PCM F32 样本。
    /// newText 中每段录音对应一个 MTMDContext.mediaMarker。
    /// mtmd 模型必须 supportsAudio 才能生效,否则抛错。
    func generateIncrementalResponseWithAudio(
        newText: String,
        audioSamples: [[Float]],
        onToken: @escaping (String) -> Bool
    ) async throws {
        guard let mtmd = mtmd, mtmd.supportsAudio else {
            throw NSError(domain: "LlamaContext", code: 17, userInfo: [NSLocalizedDescriptionKey: "当前 mmproj 不支持音频"])
        }
        let bitmaps = audioSamples.map { MTMDBitmap(pcmF32: $0) }
        try await generateIncrementalResponseWithMedia(newText: newText, bitmaps: bitmaps, onToken: onToken)
    }

    /// 图 + 音混合路径。newText 里标记顺序必须严格先图后音,和 bitmaps 顺序对齐。
    func generateIncrementalResponseWithMixedMedia(
        newText: String,
        images: [Data],
        audioSamples: [[Float]],
        onToken: @escaping (String) -> Bool
    ) async throws {
        let imageBitmaps: [MTMDBitmap] = images.compactMap { MTMDBitmap.fromImageData($0) }
        guard imageBitmaps.count == images.count else {
            throw NSError(domain: "LlamaContext", code: 11, userInfo: [NSLocalizedDescriptionKey: "图片解码失败"])
        }
        let audioBitmaps = audioSamples.map { MTMDBitmap(pcmF32: $0) }
        try await generateIncrementalResponseWithMedia(newText: newText, bitmaps: imageBitmaps + audioBitmaps, onToken: onToken)
    }

    /// 共享实现:接收已经构造好的 MTMDBitmap 数组 (图像或音频都行),
    /// 用 mtmd_tokenize 拆 chunks,再把 chunks 喂进 llama KV cache,
    /// 最后进入共享的 sampleLoop 出字。
    private func generateIncrementalResponseWithMedia(
        newText: String,
        bitmaps: [MTMDBitmap],
        onToken: @escaping (String) -> Bool
    ) async throws {
        guard let model = model, let ctx = ctx, let mtmd = mtmd else {
            throw NSError(domain: "LlamaContext", code: 10, userInfo: [NSLocalizedDescriptionKey: "mtmd 未初始化"])
        }

        // ⛔️ 进入新一轮生成前清掉上一轮遗留的取消信号
        cancelFlag.reset()

        let chunks = try mtmd.tokenize(prompt: newText, bitmaps: bitmaps, addSpecial: (n_past == 0))
        let n_embd = Int(llama_model_n_embd(model))

        // 预算一下总 token 数,空间不足就重置 (和文字路径保持一致)
        var plannedTokens: Int = 0
        for i in 0..<chunks.count { plannedTokens += chunks[i].nTokens }
        if n_past + Int32(plannedTokens) + 1024 > n_ctx {
            print("⚠️ 带媒体的 prompt 剩余上下文不足,正在重置...")
            self.reset()
            try await generateIncrementalResponseWithMedia(newText: newText, bitmaps: bitmaps, onToken: onToken)
            return
        }

        self.batch.n_tokens = 0
        for i in 0..<chunks.count {
            // ⛔️ 多模态 chunk 处理很重 (image encode 动辄数百 ms) — 每 chunk 开头查一次
            if cancelFlag.isCancelled {
                print("⛔️ 多模态 chunk 处理中途收到取消信号")
                return
            }

            let chunk = chunks[i]
            let isFinalChunk = (i == chunks.count - 1)

            switch chunk.type {
            case .text:
                let tokens = chunk.textTokens
                for (j, tok) in tokens.enumerated() {
                    let isLast = isFinalChunk && (j == tokens.count - 1)
                    batch_add(token: tok, pos: n_past + Int32(j), logits: isLast)
                    if self.batch.n_tokens >= self.safeBatchSize && !isLast {
                        if llama_decode(ctx, batch) != 0 {
                            throw NSError(domain: "LlamaContext", code: 12, userInfo: [NSLocalizedDescriptionKey: "文字 chunk decode 失败"])
                        }
                        self.batch.n_tokens = 0
                        await Task.yield()
                    }
                }
                if !isFinalChunk && self.batch.n_tokens > 0 {
                    if llama_decode(ctx, batch) != 0 {
                        throw NSError(domain: "LlamaContext", code: 13, userInfo: [NSLocalizedDescriptionKey: "文字 chunk 收尾 decode 失败"])
                    }
                    self.batch.n_tokens = 0
                }
                n_past += Int32(tokens.count)

            case .image, .audio:
                // 把之前没 flush 的文字 batch 先送掉,保证 KV 是连续的
                if self.batch.n_tokens > 0 {
                    if llama_decode(ctx, batch) != 0 {
                        throw NSError(domain: "LlamaContext", code: 14, userInfo: [NSLocalizedDescriptionKey: "image 前 flush decode 失败"])
                    }
                    self.batch.n_tokens = 0
                }

                try mtmd.encode(chunk: chunk)
                guard let embdSrc = mtmd.outputEmbedPointer() else {
                    throw NSError(domain: "LlamaContext", code: 15, userInfo: [NSLocalizedDescriptionKey: "mtmd 输出 embd 为空"])
                }
                let nTok = chunk.nTokens
                var embdBatch = llama_batch_init(Int32(nTok), Int32(n_embd), 1)
                embdBatch.n_tokens = Int32(nTok)
                let byteCount = n_embd * nTok * MemoryLayout<Float>.size
                memcpy(embdBatch.embd, embdSrc, byteCount)
                for j in 0..<nTok {
                    embdBatch.pos[j] = n_past + Int32(j)
                    embdBatch.n_seq_id[j] = 1
                    embdBatch.seq_id[j]?[0] = 0
                    embdBatch.logits[j] = (isFinalChunk && j == nTok - 1) ? 1 : 0
                }
                let rc = llama_decode(ctx, embdBatch)
                llama_batch_free(embdBatch)
                if rc != 0 {
                    throw NSError(domain: "LlamaContext", code: 16, userInfo: [NSLocalizedDescriptionKey: "embd decode 失败"])
                }
                n_past += Int32(chunk.nPos)
                await Task.yield()

                // 最后一个 chunk 是图片的话,需要把 token 位置留给采样用的 batch
                if isFinalChunk {
                    // 给 sampleLoop 准备一个新 token 的空 batch
                    // 实际上 sampleLoop 第一轮的 llama_decode 会用 self.batch,
                    // 这里塞一个哨兵 token 反而会出错。改为让 sampleLoop 直接
                    // 从 logits[n_past-1] 读,构造出下一个 batch。
                    // 做法:self.batch.n_tokens 已是 0,sampleLoop 里首轮
                    //      llama_decode 会空跑 —— 所以我们不能直接进 sampleLoop。
                    // 取巧:sampleLoop 第一步是 llama_decode(ctx, batch),
                    //       batch 是空的会崩。需要先手工 sample 一次第一 token,
                    //       或者修改 sampleLoop。这里选修改 sampleLoop:接受一个
                    //       "已经 decode 完,直接从 ctx 最新 logits 开始" 的模式。
                    // -> 改为 sampleLoop 开头:如果 batch.n_tokens == 0,先从
                    //    llama_get_logits_ith(ctx, -1) 取最后一位 logits 继续。
                    // 已在 sampleLoop 内实现。
                }
            }
        }

        try await sampleLoop(model: model, ctx: ctx, onToken: onToken)
    }

    // MARK: - 🎲 共享采样循环
    // 前提:调用前 self.batch 已经填好带 logits=1 的最后一个 prompt token,
    //        或者已经有一次 llama_decode 是带 logits 请求的 embd/token batch
    //        (此时 self.batch.n_tokens 可能为 0,直接走"读上一次 ctx logits")。
    private func sampleLoop(model: OpaquePointer, ctx: OpaquePointer, onToken: @escaping (String) -> Bool) async throws {
        let vocab = llama_model_get_vocab(model)
        let n_vocab = llama_vocab_n_tokens(vocab)
        let max_gen_capacity = n_ctx - n_past
        var byteBuffer: [UInt8] = []

        // 决定第一次从哪里取 logits:
        //   - 有 pending batch (文字路径刚加了最后一个 token): 先 decode 一次再读 batch 末尾
        //   - 无 pending batch (图片路径,embd batch 已经 decode 过): 直接读 ctx 最新 logits
        var needInitialDecode = (self.batch.n_tokens > 0)
        var logitsIndex: Int32 = 0  // 对 llama_get_logits_ith 的参数

        for _ in 0..<max_gen_capacity {
            // ⛔️ P0-1 硬终止:每轮采样开头检查取消信号,立即退出循环并让 GPU 停下来
            if cancelFlag.isCancelled {
                print("⛔️ 采样循环收到取消信号,停止生成")
                break
            }

            if needInitialDecode {
                if llama_decode(ctx, batch) != 0 {
                    print("❌ Decode 失败")
                    break
                }
                logitsIndex = batch.n_tokens - 1
                needInitialDecode = false
            } else {
                logitsIndex = -1  // -1 = last token of previous decode
            }

            await Task.yield()

            guard let logits = llama_get_logits_ith(ctx, logitsIndex) else { break }

            // 🎯 P1-2a 官方采样参数 (Gemma 4 README "Best Practices"):
            //    temperature = 1.0
            //    top_k = 64
            //    top_p = 0.95
            // P0-24:任务式调用 (如 ASR 转写) 走低温接近 greedy 的解码 — 通过
            //        samplingOverride 临时覆盖,避免 1.0 randomness 让 ASR 创造性补字。
            let temperature: Float = samplingOverride?.temperature ?? 1.0
            let topK: Int = samplingOverride?.topK ?? 64
            let topP: Float = samplingOverride?.topP ?? 0.95
            let n_vocab_int = Int(n_vocab)

            // === 第 1 步:扫一遍找 maxLogit + 用宽阈值粗筛候选 ===
            // 阈值 maxLogit - 12 给 top_p=0.95 留充裕安全余量 (exp(-12) ≈ 6e-6)。
            // 对峰值分布只剩几十~几百候选 → 后续 sort 飞快;
            // 对均匀分布全部纳入 → 慢但正确。
            var maxLogit = logits[0]
            for i in 1..<n_vocab_int {
                if logits[i] > maxLogit { maxLogit = logits[i] }
            }
            let coarseThreshold = maxLogit - 12.0

            // (idx, logit) 对,留作后续排序
            var coarse: [(idx: Int, logit: Float)] = []
            coarse.reserveCapacity(512)
            for i in 0..<n_vocab_int {
                if logits[i] > coarseThreshold {
                    coarse.append((i, logits[i]))
                }
            }
            // 极端兜底:阈值卡得过严没拿到候选,用全局 max 直接过
            if coarse.isEmpty {
                let new_token_id_fallback = llama_token(0)
                if new_token_id_fallback == llama_vocab_eos(vocab) { break }
                self.batch.n_tokens = 0
                batch_add(token: new_token_id_fallback, pos: n_past, logits: true)
                n_past += 1
                needInitialDecode = true
                continue
            }

            // === 第 2 步:按 logit 降序排;取前 topK ===
            coarse.sort { $0.logit > $1.logit }
            let kept = Array(coarse.prefix(min(topK, coarse.count)))

            // === 第 3 步:对前 topK 做 softmax (温度 1.0 即标准 softmax) ===
            let kMax = kept[0].logit
            var probs: [Float] = kept.map { exp(($0.logit - kMax) / temperature) }
            var sumExp: Float = 0
            for p in probs { sumExp += p }
            if sumExp > 0 {
                for i in 0..<probs.count { probs[i] /= sumExp }
            }

            // === 第 4 步:top-p (nucleus) 累积截断 ===
            // 从概率最高开始累加,刚跨过 0.95 就停;保留前 cutoff 个候选。
            var cumsum: Float = 0
            var cutoff = probs.count
            for i in 0..<probs.count {
                cumsum += probs[i]
                if cumsum >= topP {
                    cutoff = i + 1
                    break
                }
            }

            // === 第 5 步:在 nucleus 内归一化后多项式采样 ===
            var nucleusSum: Float = 0
            for i in 0..<cutoff { nucleusSum += probs[i] }
            let r = Float.random(in: 0..<max(nucleusSum, 1e-9))
            var acc: Float = 0
            var pickedIdx = 0
            for i in 0..<cutoff {
                acc += probs[i]
                if r <= acc {
                    pickedIdx = i
                    break
                }
            }
            let new_token_id: llama_token = llama_token(kept[pickedIdx].idx)

            if new_token_id == llama_vocab_eos(vocab) || new_token_id == 106 || new_token_id == 212 || new_token_id == 50 {
                break
            }

            let newBytes = decodeTokenBytes(token: new_token_id)
            byteBuffer.append(contentsOf: newBytes)

            if let validString = String(bytes: byteBuffer, encoding: .utf8) {
                var piece = validString
                piece = piece.replacingOccurrences(of: "<end_of_turn>", with: "")
                piece = piece.replacingOccurrences(of: "<turn|>", with: "")
                piece = piece.replacingOccurrences(of: "</s>", with: "")

                if !piece.isEmpty {
                    let shouldContinue = onToken(piece)
                    if !shouldContinue { break }
                }
                byteBuffer.removeAll()

            } else if byteBuffer.count >= 15 {
                let fallback = String(decoding: byteBuffer, as: UTF8.self)
                var piece = fallback
                piece = piece.replacingOccurrences(of: "<end_of_turn>", with: "")
                if !piece.isEmpty {
                    let shouldContinue = onToken(piece)
                    if !shouldContinue { break }
                }
                byteBuffer.removeAll()
            }

            self.batch.n_tokens = 0
            batch_add(token: new_token_id, pos: n_past, logits: true)
            n_past += 1
            needInitialDecode = true
        }
    }
    
    func reset() {
        if let oldCtx = self.ctx {
            llama_free(oldCtx)
        }
        
        var ctx_params = llama_context_default_params()
        
        ctx_params.n_ctx = .init(self.n_ctx)
        ctx_params.n_batch = .init(self.safeBatchSize)
        
        // 🚀 性能优化：重置时也要保持多线程开足马力
        ctx_params.n_threads = 4
        ctx_params.n_threads_batch = 4
        
        self.ctx = llama_init_from_model(self.model, ctx_params)
        self.n_past = 0
        // 重置上下文时顺手清掉可能残留的取消标记,避免下一次生成一开始就被老信号打断
        cancelFlag.reset()
        print("🔄 记忆已重置")
    }
    
    private func batch_add(token: llama_token, pos: Int32, logits: Bool) {
        let i = Int(batch.n_tokens)
        batch.token[i] = token
        
        batch.pos[i] = .init(pos)
        batch.n_seq_id[i] = 1
        batch.seq_id[i]?[0] = 0
        batch.logits[i] = .init(logits ? 1 : 0)
        
        batch.n_tokens += 1
    }
    
    private func tokenize(text: String, add_bos: Bool) -> [llama_token] {
        let utf8Count = text.utf8.count
        let n_tokens = utf8Count + (add_bos ? 1 : 0) + 1
        var tokens = [llama_token](repeating: 0, count: n_tokens)
        let tokenCount = llama_tokenize(llama_model_get_vocab(model), text, Int32(utf8Count), &tokens, Int32(n_tokens), add_bos, true)
        return tokenCount < 0 ? [] : Array(tokens.prefix(Int(tokenCount)))
    }
    
    private func decodeTokenBytes(token: llama_token) -> [UInt8] {
        var buf = [CChar](repeating: 0, count: 256)
        let n = llama_token_to_piece(llama_model_get_vocab(model), token, &buf, Int32(buf.count), 0, true)
        if n > 0 {
            return buf.prefix(Int(n)).map { UInt8(bitPattern: $0) }
        }
        return []
    }
    
    deinit {
        llama_batch_free(batch)
        if let currentCtx = ctx { llama_free(currentCtx) }
        if let currentModel = model { llama_model_free(currentModel) }
        llama_backend_free()
    }
}
