package(default_visibility = ["//visibility:public"])

cc_library(
    name = "vhacd",
    srcs = glob([
        "src/*.cpp",
    ]),
    hdrs = glob([
        "src/*.h",
        "inc/*.h",
    ]),
    includes = [
        "src",
        "inc",
    ],
    copts = ["-std=c++17"],
)
