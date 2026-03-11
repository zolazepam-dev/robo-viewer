load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "robot",
    srcs = [
        "Robot.cpp",
        "RobotFactory.cpp",
        "RobotController.cpp",
    ],
    hdrs = [
        "Robot.h",
        "RobotFactory.h",
        "RobotController.h",
    ],
    deps = [
        "//modules/common:common",
        "//modules/physics:physics",
        "@jolt//:jolt",
        "@nlohmann_json//:json",
    ],
    copts = [
        "-std=c++17",
        "-O3",
        "-mavx2",
        "-mfma",
    ],
    visibility = ["//visibility:public"],
)
