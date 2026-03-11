load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "common",
    hdrs = [
        "Types.h",
        "AlignedAllocator.h",
        "PerformanceDiagnoser.h",
    ],
    copts = [
        "-std=c++17",
        "-O3",
        "-mavx2",
        "-mfma",
    ],
    visibility = ["//visibility:public"],
)
