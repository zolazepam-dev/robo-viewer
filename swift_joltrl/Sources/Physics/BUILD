load("@rules_swift//swift:swift.bzl", "swift_library")

package(default_visibility = ["//visibility:public"])

swift_library(
    name = "Physics",
    srcs = ["PhysicsWorld.swift"],
    module_name = "Physics",
    deps = [
        "@jolt//:jolt",
        "//Sources/Physics/Shims:JoltShims",
    ],
    # Enable C++ interoperability
    copts = ["-cxx-interoperability-mode=default"],
    swiftcopts = ["-cxx-interoperability-mode=default"],
)
