load("@rules_cc//cc:defs.bzl", "cc_library")

exports_files(glob(["*.cpp"]))

cc_library(
    name = "core",
    srcs = [
        "PhysicsCore.cpp",
        "Renderer.cpp",
        "CombatRobot.cpp",
        "RobotLoader.cpp",
        "OverlayUI.cpp",
        "VectorizedEnv.cpp",
        "CombatEnv.cpp",
        "TD3Trainer.cpp",
        "SpanNetwork.cpp",
        "NeuralNetwork.cpp",
        "NeuralMath.cpp",
        "LatentMemory.cpp",
        "OpponentPool.cpp",
    ],
    hdrs = [
        "PhysicsCore.h",
        "Renderer.h",
        "CombatRobot.h",
        "RobotLoader.h",
        "OverlayUI.h",
        "VectorizedEnv.h",
        "CombatEnv.h",
        "TD3Trainer.h",
        "SpanNetwork.h",
        "NeuralNetwork.h",
        "NeuralMath.h",
        "LatentMemory.h",
        "OpponentPool.h",
        "AlignedAllocator.h",
        "LockFreeQueue.h",
    ],
    deps = [
        "@glfw",
        "@glm",
        "@glew//:glew_static",
        "@nlohmann_json//:json",
        "@jolt//:jolt",
        "@imgui//:imgui",
        "@imgui//backends:platform-glfw",
        "@imgui//backends:renderer-opengl3",
    ],
    visibility = ["//visibility:public"],
    copts = [
        "-std=c++20",
        "-O3",
        "-mavx2",
        "-mfma",
        "-march=native"
    ],
)

cc_binary(
    name = "stability_check",
    srcs = ["stability_check.cpp"],
    data = ["//robots:combat_bot.json"],
    deps = [
        ":core",
        "@jolt//:jolt",
    ],
    copts = [
        "-std=c++20",
        "-O3",
        "-mavx2",
        "-mfma",
        "-march=native"
    ],
)
