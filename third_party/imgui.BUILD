load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "imgui",
    visibility = ["//visibility:public"],
    deps = ["@imgui//:imgui"],
)

cc_library(
    name = "imgui_impl_glfw",
    srcs = ["@imgui//:backends/imgui_impl_glfw.cpp"],
    hdrs = ["@imgui//:backends/imgui_impl_glfw.h"],
    includes = ["backends"],
    deps = [
        ":imgui",
        "@glfw",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "imgui_impl_opengl3",
    srcs = ["@imgui//:backends/imgui_impl_opengl3.cpp"],
    hdrs = ["@imgui//:backends/imgui_impl_opengl3.h"],
    includes = ["backends"],
    deps = [
        ":imgui",
        "@glew//:glew_static",
    ],
    visibility = ["//visibility:public"],
)
