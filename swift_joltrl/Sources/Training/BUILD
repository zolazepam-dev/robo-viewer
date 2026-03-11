load("@rules_swift//swift:swift.bzl", "swift_library")

package(default_visibility = ["//visibility:public"])

swift_library(
    name = "Training",
    srcs = [
        "TD3Trainer.swift",
        "ReplayBuffer.swift",
    ],
    module_name = "Training",
    deps = [
        "//Sources/Neural:Neural",
    ],
)
