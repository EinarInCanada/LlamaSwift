# LlamaSwift

A minimal Swift Package that bridges [llama.cpp](https://github.com/ggerganov/llama.cpp) into Swift projects on iOS and macOS.

LlamaSwift re-exports the compiled `llama` C library so Swift callers can use the full llama.cpp API without any Objective-C bridging headers or manual module maps. It is the inference backbone of [Gemholic](https://github.com/EinarInCanada/Gemholic).

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

llama.cpp is a C/C++ library. To use it in a Swift project you need:

1. A compiled `.xcframework` containing the static library and headers for each target (iOS device, iOS Simulator, macOS).
2. A Swift Package that wraps that `.xcframework` as a `binaryTarget` and re-exports it so Swift code can `import LlamaSwift` directly.

LlamaSwift does exactly step 2. Step 1 (building the xcframework) is handled by the build scripts inside the llama.cpp submodule.

```
Dependencies/
├── llama.cpp/                   # llama.cpp source + build scripts
│   └── build-apple/
│       └── llama.xcframework   # compiled output (not committed to git)
└── LlamaSwift/                  # this package
    ├── Package.swift
    └── Sources/LlamaSwift/
        └── Exports.swift        # @_exported import llama
```

---

## Installation

### Inside an Xcode project (local package)

1. Build the xcframework first (from the `llama.cpp` directory):

```bash
bash build-xcframework.sh
```

2. In Xcode, go to **File › Add Package Dependencies › Add Local…** and select the `LlamaSwift` folder.

3. Import in Swift:

```swift
import LlamaSwift

// All llama.cpp C API symbols are now available directly
let model = llama_model_load_from_file(path, params)
```

### As a dependency in another Swift Package

Because the package references a local relative path for the xcframework, it is designed for local/submodule use rather than remote SPM resolution. Clone the repo alongside your project and add it as a local package.

---

## Building the xcframework

The xcframework is not committed to this repository (it is several hundred MB). Build it once from the llama.cpp submodule:

```bash
# Standard (text-only) build
cd Dependencies/llama.cpp
bash build-xcframework.sh

# Multimodal build (includes mmproj support for vision/audio)
bash build-xcframework-mtmd.sh
```

The output is written to `Dependencies/llama.cpp/build-apple/llama.xcframework`.

---

## Updating llama.cpp

LlamaSwift pins to a specific llama.cpp commit for stability. To update:

1. Pull the desired llama.cpp commit in the submodule.
2. Rebuild the xcframework with the script above.
3. Run a full regression test on all supported iPhone models (thermal + accuracy).

---

## License

LlamaSwift is released under the [MIT License](LICENSE).

llama.cpp is copyright © 2023–2026 The ggml authors, also MIT licensed.
