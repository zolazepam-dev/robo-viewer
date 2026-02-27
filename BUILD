load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "viewer",
    srcs = ["//src:main_train.cpp"],
    data = [
        "//robots:combat_bot.json",
        "//robots:internal_bot.json",
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
    copts = ["-std=c++17", "-mavx2", "-mfma", "-O3"],
)

cc_binary(
    name = "train",
    srcs = ["//src:main_train.cpp"],
    data = [
        "//robots:combat_bot.json",
        "//robots:internal_bot.json",
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
        "-march=native",
        "-ffast-math",
        "-flto",
        "-fno-strict-aliasing"
    ],
)

cc_binary(
    name = "sequential_test",
    srcs = ["//src:sequential_test.cpp"],
    data = ["//robots:combat_bot.json"],
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
        "-march=native"
    ],
)

cc_binary(
    name = "json_test",
    srcs = ["//src:json_test.cpp"],
    data = ["//robots:combat_bot.json"],
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
    data = ["//robots:combat_bot.json"],
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
        "-march=native"
    ],
)

cc_binary(
    name = "train_headless",
    srcs = ["//src:main_train_headless.cpp"],
    data = [
        "//robots:combat_bot.json",
        "//robots:internal_bot.json",
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
        "-march=native"
    ],
)

cc_binary(
    name = "minimal_test",
    srcs = ["//src:main_train_minimal.cpp"],
    data = ["//robots:combat_bot.json"],
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
        "-march=native"
    ],
)

cc_binary(
    name = "test_json_load",
    srcs = ["//src:test_json_load.cpp"],
    data = ["//robots:combat_bot.json"],
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
        "-march=native"
    ],
)

cc_binary(
    name = "test_satellite_load",
    srcs = ["//src:test_satellite_load.cpp"],
    data = ["//robots:combat_bot.json"],
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

cc_library(
    name = "local_morphologica",
    hdrs = glob(["morphologica/morph/**/*.h", "morphologica/morph/**/*.hpp"]),
    includes = ["morphologica"],
    copts = ["-std=c++20"],
    deps = [
        "@glfw",
        "@glm",
        "@glew//:glew_static",
    ],
    linkopts = [
        "-lGL",
        "-lfreetype",
        "-lpthread",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "micro_board_simple",
    srcs = ["micro_board_simple.cpp"],
    copts = ["-std=c++17", "-O3"],
)
