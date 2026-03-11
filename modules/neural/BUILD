load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "neural",
    srcs = ["NeuralMath.cpp", "LatentMemory.cpp", "SpanNetwork.cpp"],
    hdrs = ["NeuralMath.h", "LatentMemory.h", "SpanNetwork.h"],
    deps = [
        "//modules/common:common",
    ],
    copts = ["-std=c++17", "-O3", "-mavx2", "-mfma", "-march=native"],
    visibility = ["//visibility:public"],
)
