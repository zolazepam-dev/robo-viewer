load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "jolt",
    srcs = glob(
        [
            "Jolt/**/*.cpp",
            "Jolt/**/*.inl",
        ],
        exclude = [
            "TestFramework/**",
            "Samples/**",
            "Build/**",
            "UnitTests/**",
        ],
    ),
    hdrs = glob(
        [
            "Jolt/**/*.h",
        ],
        exclude = [
            "TestFramework/**",
            "Samples/**",
            "Build/**",
            "UnitTests/**",
        ],
    ),
    includes = ["."],
    copts = [
        "-std=c++17",
    ],
    visibility = ["//visibility:public"],
)
