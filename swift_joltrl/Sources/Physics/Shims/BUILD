load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "JoltShims",
    hdrs = ["JoltShims.h"],
    deps = ["@jolt//:jolt"],
    # Ensure headers are easily discoverable
    includes = ["."],
)
