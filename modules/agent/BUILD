load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "agent",
    srcs = [
        "TD3Trainer.cpp",
        "OpponentPool.cpp",
    ],
    hdrs = [
        "TD3Trainer.h",
        "OpponentPool.h",
    ],
    deps = [
        "//modules/common:common",
        "//modules/neural:neural",
        "//modules/replay:replay",
        "@jolt//:jolt",
    ],
    copts = [
        "-std=c++17",
        "-O3",
        "-mavx2",
        "-mfma",
        "-march=native",
    ],
    visibility = ["//visibility:public"],
)
