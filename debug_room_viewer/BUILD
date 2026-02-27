load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "debug_room_viewer",
    srcs = ["main.cpp"],
    data = ["//robots:combat_bot.json"],
    deps = [
        "//src:core",
        "@glfw",
        "@glm",
        "@glew//:glew_static",
        "@jolt//:jolt",
        "@imgui//:imgui",
        "@imgui//backends:platform-glfw",
        "@imgui//backends:renderer-opengl3",
        "@nlohmann_json//:json",
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
