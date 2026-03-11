"""
Sokol single-header graphics library.
https://github.com/floooh/sokol
"""

load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "sokol",
    hdrs = [
        "sokol_gfx.h",
        "sokol_app.h",
        "sokol_glue.h",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
    # sokol_gfx.h needs OpenGL headers
    linkopts = ["-lGL"],
)
