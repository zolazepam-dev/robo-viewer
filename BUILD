load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_binary(
    name = "viewer",
    srcs = [
        "src/main.cpp",
        "src/PhysicsCore.cpp",
        "src/PhysicsCore.h",
        "src/RobotLoader.cpp",
        "src/RobotLoader.h",
        "src/Renderer.cpp",
        "src/Renderer.h",
    ],
    data = ["robots/test_bot.json"],
    deps = [
        "@glfw",
        "@glm",
        "@glew//:glew_static",
        "@nlohmann_json//:json",
        "@jolt//:jolt",
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
        # Swapped out NeuralNetwork for the new SPAN/ODE2VAE stack
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
    # MUST have -flto in linkopts for the linker to actually perform the optimization
    linkopts = ["-lGL", "-lpthread", "-flto"],
    copts = [
        "-std=c++20",
        "-O3",
        "-mavx2",
        "-mfma",
        "-march=native",
        "-ffast-math",
        "-flto",
        "-fno-strict-aliasing" # Required for the AVX2 fast-math pointer casting
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