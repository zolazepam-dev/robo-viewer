load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "environment",
    srcs = [
        "CombatEnv.cpp",
    ],
    hdrs = [
        "IEnvironment.h",
        "CombatEnv.h",
    ],
    deps = [
        "//modules/common:common",
        "//modules/physics:physics",
        "//modules/robot:robot",
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
