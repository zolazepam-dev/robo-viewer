load("@rules_cc//cc:defs.bzl", "cc_library")

cc_binary(
    name = "mjcf_viewer",
    srcs = ["mjcf_viewer.cpp"],
    deps = [
        ":core",
        "@glfw",
        "@glm",
        "@glew//:glew_static",
        "@nlohmann_json//:json",
        "@jolt//:jolt",
        "@imgui//:imgui",
        "@imgui//backends:platform-glfw",
        "@imgui//backends:renderer-opengl3",
        "@tinyxml2//:tinyxml2",
        "@assimp//:assimp",
    ],
    copts = [
        "-std=c++17",
        "-O3",
        "-mavx2",
        "-mfma",
        "-DJPH_DEBUG_RENDERER",
        "-march=native",
    ],
    linkopts = [
        "-lGL",
        "-lpthread",
    ],
)

exports_files(glob(["*.cpp"]))

cc_library(
    name = "core",
    srcs = [
        "PhysicsCore.cpp",
        "Renderer.cpp",
        "CombatRobot.cpp",
        "RobotLoader.cpp",
        "RobotTemplates.cpp",
        "OverlayUI.cpp",
        "OverlayUI_refactor.cpp",
        "VectorizedEnv.cpp",
        "CombatEnv.cpp",
        "TD3Trainer.cpp",
        "SpanNetwork.cpp",
        "RFFNetwork.cpp",
        "RFFLayer.cpp",
        "RFFLatentDynamics.cpp",
        "NeuralNetwork.cpp",
        "NeuralMath.cpp",
        "LatentMemory.cpp",
        "OpponentPool.cpp",
        "InternalRobot.cpp",
        "Aircraft.cpp",
        "AircraftEnv.cpp",
        "RobotConfig.cpp",
        "RobotFactory.cpp",
        "RobotController.cpp",
        "OctopodLoader.cpp",
        "OctopodEnv.cpp",
        "VectorizedOctopodEnv.cpp",
        "ShardEnv.cpp",
        "MuonOptimizer.cpp",
        "ParallelPhysics.cpp",
        "ConfigManager.cpp",
        "Tensor.cpp",
        "TensorMath.cpp",
    ],
    hdrs = [
        "PhysicsCore.h",
        "Renderer.h",
        "CombatRobot.h",
        "RobotLoader.h",
        "RobotTemplates.h",
        "OverlayUI.h",
        "OverlayUI_refactor.h",
        "ConfigManager.h",
        "VectorizedEnv.h",
        "CombatEnv.h",
        "TD3Trainer.h",
        "SpanNetwork.h",
        "RFFNetwork.h",
        "RFFLayer.h",
        "RFFLatentDynamics.h",
        "NeuralNetwork.h",
        "NeuralMath.h",
        "LatentMemory.h",
        "OpponentPool.h",
        "AlignedAllocator.h",
        "LockFreeQueue.h",
        "InternalRobot.h",
        "ArmaEigenUtils.h",
        "EigenUtils.h",
        "PerformanceProfiler.h",
        "OptimizedMath.h",
        "MuonOptimizer.h",
        "SIMDObservations.h",
        "ParallelPhysics.h",
        "VisualState.h",
        "Aircraft.h",
        "AircraftEnv.h",
        "Robot.h",
        "RobotConfig.h",
        "RobotFactory.h",
        "RobotController.h",
        "OctopodLoader.h",
        "OctopodEnv.h",
        "VectorizedOctopodEnv.h",
        "ShardEnv.h",
        "RobotLoader_fixed.h",
        "Logging.h",
        "LockFreeActionBuffer.h",
        "AlignedAllocator.hpp",
        "Tensor.hpp",
        "SimdOps.hpp",
        "RFFTensor.hpp",
        "RFF.hpp",
        # Optimized training headers
        "OptimizedBatchOps.h",
        "ParallelPhysicsStepper.h",
        "SoAEnvBatch.h",
        "OptimizedTrainingPipeline.h",
        # Phase 4 SoA optimization
        "SoAEnvironment.h",
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
        "@tinyxml2//:tinyxml2",
        "@assimp//:assimp",
        ":linalg_utils",
    ],
    visibility = ["//visibility:public"],
    copts = [
        "-std=c++17",
        "-O3",
        "-mavx2",
        "-mfma",
        "-DJPH_DEBUG_RENDERER",
        "-march=native",
        "-fopenmp",
        "-I/usr/include/eigen3"
    ],
)

cc_library(
    name = "linalg_utils",
    hdrs = ["ArmaEigenUtils.h"],
    copts = [
        "-I/usr/include/eigen3",
    ],
    linkopts = ["-larmadillo"],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "aircraft_viewer",
    srcs = [
        "aircraft_viewer.cpp",
        "Aircraft.h",
        "AircraftEnv.h",
    ],
    includes = ["."],
    deps = [
        ":core",
        "@glfw",
        "@glm",
        "@glew//:glew_static",
        "@jolt//:jolt",
        "@imgui//:imgui",
        "@imgui//backends:platform-glfw",
        "@imgui//backends:renderer-opengl3",
    ],
    linkopts = ["-lGL", "-lpthread"],
    copts = ["-std=c++17", "-mavx2", "-mfma",
        "-DJPH_DEBUG_RENDERER", "-O3"],
    visibility = ["//visibility:public"],
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
        "-DJPH_DEBUG_RENDERER",
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
        "-DJPH_DEBUG_RENDERER",
        "-march=native",
    ],
)

cc_binary(
    name = "gradient_check_test",
    srcs = ["gradient_check_test.cpp"],
    deps = [
        ":core",
        "@jolt//:jolt",
    ],
    copts = [
        "-std=c++17",
        "-O3",
        "-mavx2",
        "-mfma",
        "-DJPH_DEBUG_RENDERER",
        "-march=native",
        "-I/usr/include/eigen3"
    ],
)

cc_test(
    name = "NeuralMathTest",
    srcs = ["NeuralMathTest.cpp"],
    deps = [
        ":core",
        "@googletest//:gtest_main",
    ],
    copts = [
        "-std=c++17",
        "-O3",
        "-mavx2",
        "-mfma",
        "-march=native",
        "-I/usr/include/eigen3"
    ],
    linkopts = ["-lgomp"],
)

cc_test(
    name = "RFFTensorTest",
    srcs = ["RFFTensorTest.cpp"],
    deps = [
        ":core",
        "@googletest//:gtest_main",
    ],
    copts = [
        "-std=c++17",
        "-O3",
        "-mavx2",
        "-mfma",
        "-march=native",
    ],
    linkopts = ["-lgomp"],
)

cc_test(
    name = "CriticalBugsTest",
    srcs = ["CriticalBugsTest.cpp"],
    deps = [
        ":core",
        "@googletest//:gtest_main",
    ],
    copts = [
        "-std=c++17",
        "-O3",
        "-mavx2",
        "-mfma",
        "-march=native",
        "-I/usr/include/eigen3"
    ],
    linkopts = ["-lgomp", "-lpthread"],
)

# SIMD Vectorization Test - Phase 3 of SPS Performance Optimization
# Tests AVX2/FMA optimizations for observation preprocessing, reward calculation,
# and batch environment stepping
cc_binary(
    name = "simd_vectorization_test",
    srcs = ["SIMDVectorizationTest.cpp"],
    deps = [
        ":core",
    ],
    copts = [
        "-std=c++17",
        "-O3",
        "-mavx2",
        "-mfma",
        "-march=native",
        "-ffast-math",
        "-I/usr/include/eigen3",
    ],
    linkopts = ["-lpthread"],
)
