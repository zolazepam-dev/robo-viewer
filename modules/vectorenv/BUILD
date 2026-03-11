load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "vectorenv",
    srcs = ["VectorEnv.cpp"],
    hdrs = ["VectorEnv.h"],
    deps = [
        "//modules/common:common",
        "//modules/physics:physics",
        "//modules/environment:environment",
        "@jolt//:jolt",
    ],
    copts = [
        "-std=c++17",
        "-O3",
        "-mavx2",
        "-mfma",
    ],
    visibility = ["//visibility:public"],
)
