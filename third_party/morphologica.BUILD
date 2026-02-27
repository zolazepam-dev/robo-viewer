load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "morphologica",
    hdrs = glob(["morph/**/*.h", "morph/**/*.hpp"]),
    includes = ["morph"],
    copts = ["-std=c++20"],
    deps = [
        "@glfw",
        "@glm",
        "@glew//:glew_static",
    ],
    linkopts = [
        "-lGL",
        "-lfreetype",
        "-lpthread",
    ],
    visibility = ["//visibility:public"],
)