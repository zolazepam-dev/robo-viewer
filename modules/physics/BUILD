load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "physics",
    srcs = ["PhysicsWorld.cpp"],
    hdrs = ["PhysicsWorld.h"],
    deps = [
        "//modules/common:common",
        "@jolt//:jolt",
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
