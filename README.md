# LlamaSwift

A Swift Package that brings [llama.cpp](https://github.com/ggerganov/llama.cpp) inference to iOS, including full multimodal support (images and audio) via `libmtmd`.



---

## What's included

| File | Description |
|---|---|
| `Exports.swift` | Re-exports the compiled `llama` C library into Swift |
| `LlamaContext.swift` | Full inference engine: context lifecycle, text generation, multimodal prompt feeding, custom sampler |
| `MTMDContext.swift` | Swift bridge over `libmtmd`: image/audio bitmap handling, tokenization, chunk encoding, embedding extraction |

### LlamaContext

- Loads a GGUF model and initialises a 32 K-token context window with Metal GPU acceleration
- `generateIncrementalResponse` — streaming text generation with a cancel flag
- `generateIncrementalResponseWithImages` — feeds image frames into the KV cache via libmtmd then streams tokens
- `generateIncrementalResponseWithAudio` — same for PCM F32 audio
- `generateIncrementalResponseWithMixedMedia` — images and audio in a single prompt
- Hand-written Top-K / Top-P / Temperature sampler (no external sampling library)
- `SamplingOverride` for task-mode decoding (e.g. near-greedy for ASR)
- `LlamaCancelFlag` — thread-safe hard stop; UI can call `cancel()` mid-generation

### MTMDContext

- Wraps `mtmd_context` lifecycle (init, free)
- `MTMDBitmap` — decode JPEG/PNG/HEIC bytes → RGB24 via CGContext; or wrap PCM F32 audio
- `MTMDChunks` / `MTMDChunk` — typed chunk iterator (text / image / audio) with token counts and M-RoPE position info
- `tokenize(prompt:bitmaps:)` — splits a marked-up prompt into chunks aligned to bitmap list
- `encode(chunk:)` — runs the vision/audio tower and exposes the output embedding pointer

---

## Requirements

| Requirement | Version |
|---|---|
| iOS | 16.0+ |
| Xcode | 15.0+ |
| Swift | 5.9+ |
| llama.cpp xcframework | Built separately (see below) |

---

## How it works

```
Your Swift code
    ↓  import LlamaSwift
LlamaContext / MTMDContext
    ↓  (internal)
llama.xcframework  ←  compiled from llama.cpp + libmtmd
    ↓
Gemma / llama model inference on-device (Metal)
```

The `llama.xcframework` is a local binary target — it is not committed to this repo. You build it once from the llama.cpp source tree (see below).

---

## Project layout

```
Dependencies/
├── llama.cpp/                        # llama.cpp source (git submodule)
│   ├── build-xcframework.sh          # text-only build
│   ├── build-xcframework-mtmd.sh     # multimodal build (required for images/audio)
│   └── build-apple/
│       └── llama.xcframework         # compiled output (not committed)
└── LlamaSwift/                       # this package
    ├── Package.swift
    └── Sources/LlamaSwift/
        ├── Exports.swift
        ├── LlamaContext.swift
        └── MTMDContext.swift
```

---

## Installation

### 1. Build the xcframework

```bash
cd Dependencies/llama.cpp

# For text + images + audio (recommended):
bash build-xcframework-mtmd.sh

# Text-only:
bash build-xcframework.sh
```

Output goes to `Dependencies/llama.cpp/build-apple/llama.xcframework`.

### 2. Add the local package in Xcode

**File › Add Package Dependencies › Add Local…** → select the `LlamaSwift` folder.

### 3. Use in Swift

```swift
import LlamaSwift

// Load model
let engine = try LlamaContext.create_context(path: "/path/to/model.gguf")

// Optional: load multimodal projector
engine.loadMMProj(path: "/path/to/mmproj.gguf")

// Stream a text response
try await engine.generateIncrementalResponse(newText: prompt) { token in
    print(token, terminator: "")
    return true  // return false to stop early
}

// Stream a response with an image
let imageData: Data = ...  // JPEG / PNG / HEIC bytes
try await engine.generateIncrementalResponseWithImages(
    newText: "Describe this image. \(MTMDContext.mediaMarker)",
    images: [imageData]
) { token in
    print(token, terminator: "")
    return true
}
```

---

## Updating llama.cpp

LlamaSwift pins to a specific llama.cpp commit for Metal stability. To update:

1. Pull the desired commit in the submodule.
2. Rebuild the xcframework with `build-xcframework-mtmd.sh`.
3. Run a full regression on all supported devices (thermal + output quality).

---

## License

LlamaSwift is released under the [MIT License](LICENSE).

llama.cpp is copyright © 2023–2026 The ggml authors, also MIT licensed.
