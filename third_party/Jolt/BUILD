# Jolt Physics Library
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "jolt",
    srcs = glob(["Jolt/**/*.cpp", "Jolt/**/*.inl"]),
    hdrs = glob(["Jolt/**/*.h", "Jolt/**/*.inl"]),
    includes = ["."],
    local_defines = ["JPH_DEBUG_RENDERER=1"],
    visibility = ["//visibility:public"],
)
