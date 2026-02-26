load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "viewer",
    srcs = ["//src:main_train.cpp"],
    data = ["//robots:combat_bot.json"],
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
    copts = ["-std=c++20", "-mavx2", "-mfma", "-O3"],
)

cc_binary(
    name = "train",
    srcs = ["//src:main_train.cpp"],
    data = ["//robots:combat_bot.json"],
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
    srcs = ["//src:sequential_test.cpp"],
    data = ["//robots:combat_bot.json"],
    deps = [
        "//src:core",
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
    srcs = ["//src:json_test.cpp"],
    data = ["//robots:combat_bot.json"],
    deps = [
        "@nlohmann_json//:json",
    ],
    copts = ["-std=c++20"],
)

cc_binary(
    name = "jolt_test",
    srcs = ["//src:jolt_test.cpp"],
    deps = [
        "@jolt//:jolt",
    ],
    copts = ["-std=c++20"],
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
        "-std=c++20",
        "-O3",
        "-mavx2",
        "-mfma",
        "-march=native"
    ],
)

cc_binary(
    name = "train_headless",
    srcs = ["//src:main_train_headless.cpp"],
    data = ["//robots:combat_bot.json"],
    deps = [
        "//src:core",
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
    srcs = ["//src:system_test.cpp"],
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
    srcs = ["//src:test_json_load.cpp"],
    data = ["//robots:combat_bot.json"],
    deps = [
        "@nlohmann_json//:json",
    ],
    copts = ["-std=c++20"],
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
        "-std=c++20",
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
        "-std=c++20",
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
        "-std=c++20",
        "-O3",
        "-mavx2",
        "-mfma",
        "-march=native"
    ],
)
