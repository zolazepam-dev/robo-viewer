load("@rules_swift//swift:swift.bzl", "swift_library")

package(default_visibility = ["//visibility:public"])

swift_library(
    name = "Environment",
    srcs = ["CombatEnv.swift"],
    module_name = "Environment",
    deps = [
        "//Sources/Physics:Physics",
        "//Sources/Robot:Robot",
    ],
)
