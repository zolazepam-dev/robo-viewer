load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary")

cc_library(
    name = "trainloop",
    srcs = ["TrainLoop.cpp"],
    hdrs = ["TrainLoop.h"],
    deps = [
        "//modules/common:common",
        "//modules/physics:physics",
        "//modules/vectorenv:vectorenv",
        "//modules/replay:replay",
        "//modules/visualizer:visualizer",
        "//modules/agent:agent",
        "@jolt//:jolt",
        "@nlohmann_json//:json",
    ],
    copts = [
        "-std=c++17",
        "-O3",
        "-mavx2",
        "-mfma",
    ],
    linkopts = ["-lpthread"],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "main_modular",
    srcs = ["main_modular.cpp"],
    deps = [":trainloop"],
    copts = [
        "-std=c++17",
        "-O3",
        "-mavx2",
        "-mfma",
        "-march=native",
        "-ffast-math",
    ],
    linkopts = ["-lGL", "-lpthread", "-flto"],
)

cc_binary(
    name = "main_headless_modular",
    srcs = ["main_headless_modular.cpp"],
    deps = [":trainloop"],
    copts = [
        "-std=c++17",
        "-O3",
        "-mavx2",
        "-mfma",
        "-march=native",
        "-ffast-math",
    ],
    linkopts = ["-lpthread", "-flto"],
)
