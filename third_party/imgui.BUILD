cc_library(
    name = "imgui",
    srcs = [
        "imgui.cpp",
        "imgui_draw.cpp",
        "imgui_tables.cpp",
        "imgui_widgets.cpp",
    ],
    hdrs = [
        "imgui.h",
        "imconfig.h",
        "imgui_internal.h",
        "imstb_rectpack.h",
        "imstb_textedit.h",
        "imstb_truetype.h",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "imgui_impl_glfw",
    srcs = ["backends/imgui_impl_glfw.cpp"],
    hdrs = ["backends/imgui_impl_glfw.h"],
    deps = [
        ":imgui",
        "@glfw",
    ],
    includes = ["backends", "."],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "imgui_impl_opengl3",
    srcs = [
        "backends/imgui_impl_opengl3.cpp",
        "backends/imgui_impl_opengl3_loader.h",
    ],
    hdrs = ["backends/imgui_impl_opengl3.h"],
    deps = [":imgui"],
    includes = ["backends", "."],
    visibility = ["//visibility:public"],
)
