load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_binary(
    name = "viewer",
    srcs = [
        "src/combat_main.cpp",
        "src/PhysicsCore.cpp",
        "src/PhysicsCore.h",
        "src/CombatEnv.cpp",
        "src/CombatEnv.h",
        "src/CombatRobot.cpp",
        "src/CombatRobot.h",
        "src/RobotLoader.cpp",
        "src/RobotLoader.h",
        "src/Renderer.cpp",
        "src/Renderer.h",
        "src/NeuralNetwork.cpp",
        "src/NeuralNetwork.h",
        "src/SpanNetwork.cpp",
        "src/SpanNetwork.h",
        "src/NeuralMath.cpp",
        "src/NeuralMath.h",
        "src/LatentMemory.cpp",
        "src/LatentMemory.h",
        "src/TD3Trainer.cpp",
        "src/TD3Trainer.h",
        "src/OverlayUI.cpp",
        "src/OverlayUI.h",
        "src/AlignedAllocator.h",
        "src/LockFreeQueue.h",
        "src/VectorizedEnv.cpp",
        "src/VectorizedEnv.h",
        "src/OpponentPool.cpp",
        "src/OpponentPool.h",
    ],
    data = ["robots/combat_bot.json"],
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
    linkopts = ["-lGL", "-lpthread"],
    copts = ["-std=c++20", "-mavx2", "-mfma", "-O3"],
)

cc_binary(
    name = "train",
    srcs = [
        "src/main_train.cpp",
        "src/PhysicsCore.cpp",
        "src/PhysicsCore.h",
        "src/CombatRobot.cpp",
        "src/CombatRobot.h",
        "src/CombatEnv.cpp",
        "src/CombatEnv.h",
        "src/VectorizedEnv.cpp",
        "src/VectorizedEnv.h",
        "src/NeuralNetwork.cpp",
        "src/NeuralNetwork.h",
        "src/SpanNetwork.cpp",
        "src/SpanNetwork.h",
        "src/NeuralMath.cpp",
        "src/NeuralMath.h",
        "src/LatentMemory.cpp",
        "src/LatentMemory.h",
        "src/OpponentPool.cpp",
        "src/OpponentPool.h",
        "src/TD3Trainer.cpp",
        "src/TD3Trainer.h",
        "src/Renderer.cpp",
        "src/Renderer.h",
        "src/AlignedAllocator.h",
        "src/LockFreeQueue.h",
        "src/OverlayUI.cpp",
        "src/OverlayUI.h",
    ],
    data = ["robots/combat_bot.json"],
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
    linkopts = ["-lGL", "-lpthread", "-flto"],
    copts = [
        "-std=c++20",
        "-O3",
        "-mavx2",
        "-mfma",
        "-march=native",
        "-ffast-math",
        "-flto",
        "-fno-strict-aliasing"
    ],
)

cc_binary(
    name = "sequential_test",
    srcs = [
        "src/sequential_test.cpp",
        "src/PhysicsCore.cpp",
        "src/PhysicsCore.h",
        "src/CombatRobot.cpp",
        "src/CombatRobot.h",
    ],
    data = ["robots/combat_bot.json"],
    deps = [
        "@nlohmann_json//:json",
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

cc_binary(
    name = "json_test",
    srcs = ["src/json_test.cpp"],
    data = ["robots/combat_bot.json"],
    deps = [
        "@nlohmann_json//:json",
    ],
    copts = ["-std=c++20"],
)

cc_binary(
    name = "jolt_test",
    srcs = ["src/jolt_test.cpp"],
    deps = [
        "@jolt//:jolt",
    ],
    copts = ["-std=c++20"],
)

cc_binary(
    name = "simple_test",
    srcs = [
        "src/simple_test.cpp",
        "src/PhysicsCore.cpp",
        "src/PhysicsCore.h",
        "src/CombatRobot.cpp",
        "src/CombatRobot.h",
    ],
    data = ["robots/combat_bot.json"],
    deps = [
        "@nlohmann_json//:json",
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

cc_binary(
    name = "train_headless",
    srcs = [
        "src/main_train_headless.cpp",
        "src/PhysicsCore.cpp",
        "src/PhysicsCore.h",
        "src/CombatRobot.cpp",
        "src/CombatRobot.h",
        "src/CombatEnv.cpp",
        "src/CombatEnv.h",
        "src/VectorizedEnv.cpp",
        "src/VectorizedEnv.h",
        "src/NeuralNetwork.cpp",
        "src/NeuralNetwork.h",
        "src/SpanNetwork.cpp",
        "src/SpanNetwork.h",
        "src/NeuralMath.cpp",
        "src/NeuralMath.h",
        "src/LatentMemory.cpp",
        "src/LatentMemory.h",
        "src/OpponentPool.cpp",
        "src/OpponentPool.h",
        "src/TD3Trainer.cpp",
        "src/TD3Trainer.h",
        "src/AlignedAllocator.h",
        "src/LockFreeQueue.h",
    ],
    data = ["robots/combat_bot.json"],
    deps = [
        "@nlohmann_json//:json",
        "@jolt//:jolt",
    ],
    linkopts = ["-lpthread", "-flto"],
    copts = [
        "-std=c++20",
        "-O3",
        "-mavx2",
        "-mfma",
        "-march=native",
        "-ffast-math",
        "-flto",
        "-fno-strict-aliasing"
    ],
)

cc_binary(
    name = "system_test",
    srcs = ["src/system_test.cpp"],
    copts = [
        "-std=c++20",
        "-O3",
        "-mavx2",
        "-mfma",
        "-march=native"
    ],
)

cc_binary(
    name = "test_json_load",
    srcs = ["src/test_json_load.cpp"],
    data = ["robots/combat_bot.json"],
    deps = [
        "@nlohmann_json//:json",
    ],
    copts = ["-std=c++20"],
)

cc_binary(
    name = "test_satellite_load",
    srcs = [
        "src/test_satellite_load.cpp",
        "src/PhysicsCore.cpp",
        "src/PhysicsCore.h",
        "src/CombatRobot.cpp",
        "src/CombatRobot.h",
    ],
    data = ["robots/combat_bot.json"],
    deps = [
        "@nlohmann_json//:json",
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