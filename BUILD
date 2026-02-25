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
    copts = ["-std=c++17", "-mavx2", "-mfma", "-O3"],
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
        "@imgui//:imgui_impl_glfw",
        "@imgui//:imgui_impl_opengl3",
    ],
    linkopts = ["-lGL", "-lpthread"],
    copts = ["-std=c++17", "-mavx2", "-mfma", "-O3", "-march=native", "-flto", "-ffast-math"],
)
