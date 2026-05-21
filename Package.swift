// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "LlamaSwift",
    platforms: [.iOS(.v16)],
    products: [
        .library(name: "LlamaSwift", targets: ["LlamaSwift"]),
    ],
    targets: [
        .binaryTarget(
            name: "llama",
            path: "../llama.cpp/build-apple/llama.xcframework"
        ),
        .target(
            name: "LlamaSwift",
            dependencies: ["llama"],
            path: "Sources/LlamaSwift"
        ),
    ]
)
