"""
System Vulkan library.
"""

load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "vulkan",
    hdrs = [],  # No headers in this package
    includes = ["/usr/include"],
    copts = ["-I/usr/include"],
    visibility = ["//visibility:public"],
    linkopts = ["-lvulkan"],
)
