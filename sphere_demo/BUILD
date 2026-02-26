load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "sphere_demo",
    srcs = ["main.cpp"],
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
