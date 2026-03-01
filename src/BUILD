load("@rules_cc//cc:defs.bzl", "cc_library")

exports_files(glob(["*.cpp"]))

cc_library(
    name = "core",
    srcs = [
        "PhysicsCore.cpp",
        "Config.cpp",
        "Renderer.cpp",
        "CombatRobot.cpp",
        "RobotLoader.cpp",
        "OverlayUI.cpp",
        "OverlayUI_refactor.cpp",
        "VectorizedEnv.cpp",
        "CombatEnv.cpp",
        "TD3Trainer.cpp",
        "SpanNetwork.cpp",
        "NeuralNetwork.cpp",
        "NeuralMath.cpp",
        "LatentMemory.cpp",
        "OpponentPool.cpp",
        "InternalRobot.cpp",
    ],
    hdrs = [
        "PhysicsCore.h",
        "Config.h",
        "Renderer.h",
        "CombatRobot.h",
        "RobotLoader.h",
        "OverlayUI.h",
        "OverlayUI_refactor.h",
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
        "InternalRobot.h",
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
        "-std=c++17",
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
        "-std=c++17",
        "-O3",
        "-mavx2",
        "-mfma",
        "-march=native"
    ],
)

cc_binary(
    name = "practice_xml_viewer",
    srcs = ["xml_viewer.cpp"],
    deps = [
        ":core",
        "@jolt//:jolt",
        "@glfw",
        "@glm",
        "@glew//:glew_static",
    ],
    copts = [
        "-std=c++17",
        "-O3",
        "-mavx2",
        "-mfma",
        "-march=native",
    ],
)
