load("@rules_swift//swift:swift.bzl", "swift_binary")

package(default_visibility = ["//visibility:public"])

swift_binary(
    name = "swift_train",
    srcs = ["main.swift"],
    deps = [
        "//Sources/Physics:Physics",
        "//Sources/Neural:Neural",
        "//Sources/Robot:Robot",
        "//Sources/Environment:Environment",
        "//Sources/Training:Training",
    ],
    copts = ["-cxx-interoperability-mode=default"],
    swiftcopts = ["-cxx-interoperability-mode=default"],
)
