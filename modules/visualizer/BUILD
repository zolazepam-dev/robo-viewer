load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "visualizer",
    srcs = ["Visualizer.cpp"],
    hdrs = ["Visualizer.h"],
    deps = [
        "//modules/common:common",
        "//modules/physics:physics",
        "//src:core",
        "@glfw", "@glm", "@glew//:glew_static",
        "@jolt//:jolt",
        "@imgui//:imgui", "@imgui//backends:platform-glfw",
        "@imgui//backends:renderer-opengl3",
    ],
    copts = ["-std=c++17", "-O3", "-mavx2", "-mfma"],
    linkopts = ["-lGL"],
    visibility = ["//visibility:public"],
)
