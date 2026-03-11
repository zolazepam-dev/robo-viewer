load("@rules_swift//swift:swift.bzl", "swift_library")

package(default_visibility = ["//visibility:public"])

swift_library(
    name = "Neural",
    srcs = ["SpanNetwork.swift"],
    module_name = "Neural",
    # Enable C++ interoperability if we want to use NeuralMath from C++
    copts = ["-cxx-interoperability-mode=default"],
    swiftcopts = ["-cxx-interoperability-mode=default"],
)
