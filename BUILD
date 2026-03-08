load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

refresh_compile_commands(
    name = "refresh_compile_commands",
    # Targets to index
    targets = {
        "//:train": "",
        "//:viewer": "",
        "//:train_headless": "",
        "//:system_test": "",
    },
)

cc_binary(
    name = "glfw_test",
    srcs = ["//src:glfw_test.cpp"],
    deps = [
        "@glfw",
    ],
    linkopts = ["-lGL"],
    copts = ["-std=c++17"],
)

cc_binary(
    name = "integration_test",
    srcs = ["//src:integration_test.cpp"],
    deps = [
        "//src:core",
        "@glfw",
        "@glm",
        "@glew//:glew_static",
        "@jolt//:jolt",
        "@imgui//:imgui",
        "@imgui//backends:platform-glfw",
        "@imgui//backends:renderer-opengl3",
    ],
    linkopts = ["-lGL", "-lpthread"],
    copts = ["-std=c++17"],
)

cc_binary(
    name = "vecenv_test",
    srcs = ["//src:vecenv_test.cpp"],
    deps = [
        "//src:core",
        "@glfw",
        "@glm",
        "@glew//:glew_static",
        "@jolt//:jolt",
        "@imgui//:imgui",
        "@imgui//backends:platform-glfw",
        "@imgui//backends:renderer-opengl3",
    ],
    linkopts = ["-lGL", "-lpthread"],
    copts = ["-std=c++17"],
)

cc_binary(
    name = "viewer",
    srcs = ["//src:main_train.cpp"],
    data = [
        "//robots:combat_bot.json",
        "//robots:test_bot.json",
        "//robots:f22.json",
        "//robots:orbital_shard.json",
    ],
    deps = [
        "//src:core",
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
    copts = ["-std=c++17", "-mavx2", "-mfma",
        "-DJPH_DEBUG_RENDERER", "-O3"],
)

cc_binary(
    name = "hybrid_viewer",
    srcs = ["//src:hybrid_viewer.cpp"],
    data = [
        "//robots:combat_bot.json",
        "//robots:test_bot.json",
        "//robots:f22.json",
        "//robots:orbital_shard.json",
        "//robots:bouncy_orbiter.json",
    ],
    deps = [
        "//src:core",
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
    copts = ["-std=c++17", "-mavx2", "-mfma",
        "-DJPH_DEBUG_RENDERER", "-O3"],
)

cc_binary(
    name = "train",
    srcs = ["//src:main_train.cpp"],
    data = [
        "//robots:combat_bot.json",
        "//robots:test_bot.json",
        "//robots:f22.json",
        "//robots:orbital_shard.json",
    ],
    deps = [
        "//src:core",
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
        "-std=c++17",
        "-O3",
        "-mavx2",
        "-mfma",
        "-DJPH_DEBUG_RENDERER",
        "-march=native",
        "-ffast-math",
        "-flto",
        "-fno-strict-aliasing"
    ],
)

cc_binary(
    name = "sequential_test",
    srcs = ["//src:sequential_test.cpp"],
    data = [
        "//robots:combat_bot.json",
        "//robots:test_bot.json",
        "//robots:f22.json",
        "//robots:orbital_shard.json",
    ],
    deps = [
        "//src:core",
        "@nlohmann_json//:json",
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
    name = "json_test",
    srcs = ["//src:json_test.cpp"],
    data = [
        "//robots:combat_bot.json",
        "//robots:test_bot.json",
        "//robots:f22.json",
        "//robots:orbital_shard.json",
    ],
    deps = [
        "@nlohmann_json//:json",
    ],
    copts = ["-std=c++17"],
)

cc_binary(
    name = "jolt_test",
    srcs = ["//src:jolt_test.cpp"],
    deps = [
        "@jolt//:jolt",
    ],
    copts = ["-std=c++17"],
)

cc_binary(
    name = "simple_test",
    srcs = ["//src:simple_test.cpp"],
    data = [
        "//robots:combat_bot.json",
        "//robots:test_bot.json",
        "//robots:f22.json",
        "//robots:orbital_shard.json",
    ],
    deps = [
        "//src:core",
        "@nlohmann_json//:json",
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
    name = "train_headless",
    srcs = ["//src:main_train_headless.cpp"],
    data = [
        "//robots:combat_bot.json",
        "//robots:test_bot.json",
        "//robots:f22.json",
        "//robots:orbital_shard.json",
    ],
    deps = [
        "//src:core",
        "@nlohmann_json//:json",
        "@jolt//:jolt",
    ],
    linkopts = ["-lpthread", "-flto"],
    copts = [
        "-std=c++17",
        "-O3",
        "-mavx2",
        "-mfma",
        "-DJPH_DEBUG_RENDERER",
        "-march=native",
        "-ffast-math",
        "-flto",
        "-fno-strict-aliasing"
    ],
)

cc_binary(
    name = "system_test",
    srcs = ["//src:system_test.cpp"],
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
    name = "minimal_test",
    srcs = ["//src:main_train_minimal.cpp"],
    data = [
        "//robots:combat_bot.json",
        "//robots:test_bot.json",
        "//robots:f22.json",
        "//robots:orbital_shard.json",
    ],
    deps = [
        "//src:core",
        "@nlohmann_json//:json",
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
    name = "end_to_end_test",
    srcs = ["//src:end_to_end_test.cpp"],
    data = [
        "//robots:combat_bot.json",
        "//robots:test_bot.json",
        "//robots:f22.json",
        "//robots:orbital_shard.json",
    ],
    deps = [
        "//src:core",
        "@nlohmann_json//:json",
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
    name = "test_json_load",
    srcs = ["//src:test_json_load.cpp"],
    data = [
        "//robots:combat_bot.json",
        "//robots:test_bot.json",
        "//robots:f22.json",
        "//robots:orbital_shard.json",
    ],
    deps = [
        "@nlohmann_json//:json",
    ],
    copts = ["-std=c++17"],
)

cc_binary(
    name = "viewer_sphere",
    srcs = ["//src:viewer_sphere.cpp"],
    deps = [
        "//src:core",
        "@glfw",
        "@glm",
        "@glew//:glew_static",
        "@jolt//:jolt",
        "@imgui//:imgui",
        "@imgui//backends:platform-glfw",
        "@imgui//backends:renderer-opengl3",
    ],
    linkopts = ["-lGL", "-lpthread"],
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
    name = "simple_sphere",
    srcs = ["//src:simple_sphere.cpp"],
    deps = [
        "//src:core",
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
    name = "test_satellite_load",
    srcs = ["//src:test_satellite_load.cpp"],
    data = [
        "//robots:combat_bot.json",
        "//robots:test_bot.json",
        "//robots:f22.json",
        "//robots:orbital_shard.json",
    ],
    deps = [
        "//src:core",
        "@nlohmann_json//:json",
        "@jolt//:jolt",
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
    name = "micro_board",
    srcs = [
        "micro_board.cpp",
        "fonts/ttf-bitstream-vera/Vera.ttf",
        "fonts/ttf-bitstream-vera/VeraIt.ttf",
        "fonts/ttf-bitstream-vera/VeraBd.ttf",
        "fonts/ttf-bitstream-vera/VeraBI.ttf",
        "fonts/ttf-bitstream-vera/VeraMono.ttf",
        "fonts/ttf-bitstream-vera/VeraMoIt.ttf",
        "fonts/ttf-bitstream-vera/VeraMoBd.ttf",
        "fonts/ttf-bitstream-vera/VeraMoBI.ttf",
        "fonts/ttf-bitstream-vera/VeraSe.ttf",
        "fonts/ttf-bitstream-vera/VeraSeBd.ttf",
        "fonts/dejavu/DejaVuSans.ttf",
        "fonts/dejavu/DejaVuSans-Oblique.ttf",
        "fonts/dejavu/DejaVuSans-Bold.ttf",
        "fonts/dejavu/DejaVuSans-BoldOblique.ttf",
    ],
    deps = [
        ":local_morphologica",
        "@glfw",
        "@glm",
        "@glew//:glew_static",
    ],
    linkopts = ["-lGL", "-lfreetype", "-lpthread"],
    copts = [
        "-std=c++20",
        "-O3",
        "-I/usr/include/freetype2",
        "-Imorphologica",
        "-DMORPH_FONTS_DIR=\\\"fonts\\\"",
    ],
)



cc_binary(
    name = "micro_board_simple",
    srcs = ["micro_board_simple.cpp"],
    copts = ["-std=c++17", "-O3"],
)

cc_binary(
    name = "mjcf_viewer",
    srcs = ["//src:mjcf_viewer.cpp"],
    data = [
        "//robots:combat_bot.json",
        "//robots:test_bot.json",
        "//robots:f22.json",
        "//robots:orbital_shard.json",
    ],
    deps = [
        "//src:core",
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
    copts = [
        "-std=c++17",
        "-O3",
        "-mavx2",
        "-mfma",
        "-DJPH_DEBUG_RENDERER",
        "-march=native"
    ],
)

# ============================================================================
# Modular architecture targets
# ============================================================================

alias(name = "mod_common", actual = "//modules/common:common")
alias(name = "mod_physics", actual = "//modules/physics:physics")
alias(name = "mod_robot", actual = "//modules/robot:robot")
alias(name = "mod_environment", actual = "//modules/environment:environment")
alias(name = "mod_vectorenv", actual = "//modules/vectorenv:vectorenv")
alias(name = "mod_replay", actual = "//modules/replay:replay")
alias(name = "mod_neural", actual = "//modules/neural:neural")
alias(name = "mod_agent", actual = "//modules/agent:agent")
alias(name = "mod_visualizer", actual = "//modules/visualizer:visualizer")
alias(name = "mod_trainloop", actual = "//modules/trainloop:trainloop")

cc_binary(
    name = "train_modular",
    srcs = ["//modules/trainloop:main_modular.cpp"],
    deps = ["//modules/trainloop:trainloop"],
    linkopts = ["-lGL", "-lpthread", "-flto"],
    copts = [
        "-std=c++17", "-O3", "-mavx2", "-mfma",
        "-DJPH_DEBUG_RENDERER", "-march=native",
        "-ffast-math", "-flto", "-fno-strict-aliasing",
    ],
)

cc_binary(
    name = "train_headless_modular",
    srcs = ["//modules/trainloop:main_headless_modular.cpp"],
    deps = ["//modules/trainloop:trainloop"],
    linkopts = ["-lpthread", "-flto"],
    copts = [
        "-std=c++17", "-O3", "-mavx2", "-mfma",
        "-DJPH_DEBUG_RENDERER", "-march=native",
        "-ffast-math", "-flto", "-fno-strict-aliasing",
    ],
)

# Octopod combat training
cc_binary(
    name = "train_octopod",
    srcs = ["//src:main_train_octopod.cpp"],
    data = [
        "//robots:octopod.json",
    ],
    deps = [
        "//src:core",
        "@nlohmann_json//:json",
        "@jolt//:jolt",
    ],
    linkopts = ["-lpthread", "-flto"],
    copts = [
        "-std=c++17",
        "-O3",
        "-mavx2",
        "-mfma",
        "-DJPH_DEBUG_RENDERER",
        "-march=native",
        "-ffast-math",
        "-flto",
        "-fno-strict-aliasing"
    ],
)

cc_binary(
    name = "octopod_viewer",
    srcs = ["//src:octopod_viewer.cpp"],
    data = [
        "//robots:octopod.json",
    ],
    deps = [
        "//src:core",
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
    copts = ["-std=c++17", "-mavx2", "-mfma",
        "-DJPH_DEBUG_RENDERER", "-O3"],
)

cc_binary(
    name = "octopod_gl_viewer",
    srcs = ["//src:octopod_gl_viewer.cpp"],
    data = [
        "//robots:octopod.json",
    ],
    deps = [
        "//src:core",
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
    name = "octopod_hybrid_viewer",
    srcs = ["//src:octopod_hybrid_viewer.cpp"],
    data = [
        "//robots:octopod.json",
    ],
    deps = [
        "//src:core",
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
    name = "train_shard",
    srcs = ["//src:main_train_shard.cpp"],
    data = [
        "//robots:orbital_shard.json",
    ],
    deps = [
        "//src:core",
        "@jolt//:jolt",
    ],
    copts = ["-std=c++17", "-mavx2", "-mfma", "-O3"],
)

cc_binary(
    name = "shard_viewer",
    srcs = ["//src:shard_viewer.cpp"],
    data = [
        "//robots:orbital_shard.json",
    ],
    deps = [
        "//src:core",
        "@glfw",
        "@glm",
        "@glew//:glew_static",
        "@nlohmann_json//:json",
        "@jolt//:jolt",
    ],
    linkopts = ["-lGL", "-lpthread"],
    copts = ["-std=c++17", "-mavx2", "-mfma", "-O3"],
)
