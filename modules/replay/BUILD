load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "replay",
    srcs = ["ReplayBuffer.cpp"],
    hdrs = ["ReplayBuffer.h"],
    deps = [
        "//modules/common:common",
        "//modules/environment:environment",
    ],
    copts = ["-std=c++17", "-O3", "-I/usr/include/eigen3"],
    visibility = ["//visibility:public"],
)
