# Performance test rules for robo-viewer

load("@bazel_skylib//rules:common_settings.bzl", "string_flag")

# Flag to control performance test duration
string_flag(
    name = "perf_duration",
    build_setting_default = "10",
    visibility = ["//visibility:public"],
)

# Shell-based performance tests
sh_test(
    name = "build_time_benchmark_train",
    srcs = ["perf_test.sh"],
    args = [
        "--target", "//:train",
        "--duration", "$(perf_duration)",
        "--test-type", "build"
    ],
    timeout = "moderate",
    tags = ["performance", "build"],
    visibility = ["//visibility:public"],
)

sh_test(
    name = "build_time_benchmark_hybrid_viewer",
    srcs = ["perf_test.sh"],
    args = [
        "--target", "//:hybrid_viewer",
        "--duration", "$(perf_duration)",
        "--test-type", "build"
    ],
    timeout = "moderate",
    tags = ["performance", "build"],
    visibility = ["//visibility:public"],
)

sh_test(
    name = "runtime_perf_train",
    srcs = ["perf_test.sh"],
    args = [
        "--target", "//:train",
        "--duration", "$(perf_duration)",
        "--test-type", "runtime"
    ],
    timeout = "moderate",
    tags = ["performance", "runtime"],
    visibility = ["//visibility:public"],
)

sh_test(
    name = "runtime_perf_hybrid_viewer",
    srcs = ["perf_test.sh"],
    args = [
        "--target", "//:hybrid_viewer",
        "--duration", "$(perf_duration)",
        "--test-type", "runtime"
    ],
    timeout = "moderate",
    tags = ["performance", "runtime"],
    visibility = ["//visibility:public"],
)

sh_test(
    name = "memory_usage_analysis_train",
    srcs = ["perf_test.sh"],
    args = [
        "--target", "//:train",
        "--duration", "$(perf_duration)",
        "--test-type", "memory"
    ],
    timeout = "moderate",
    tags = ["performance", "memory"],
    visibility = ["//visibility:public"],
)

sh_test(
    name = "memory_usage_analysis_hybrid_viewer",
    srcs = ["perf_test.sh"],
    args = [
        "--target", "//:hybrid_viewer",
        "--duration", "$(perf_duration)",
        "--test-type", "memory"
    ],
    timeout = "moderate",
    tags = ["performance", "memory"],
    visibility = ["//visibility:public"],
)

sh_test(
    name = "cpu_utilization_monitor_train",
    srcs = ["perf_test.sh"],
    args = [
        "--target", "//:train",
        "--duration", "$(perf_duration)",
        "--test-type", "cpu"
    ],
    timeout = "moderate",
    tags = ["performance", "cpu"],
    visibility = ["//visibility:public"],
)

sh_test(
    name = "cpu_utilization_monitor_hybrid_viewer",
    srcs = ["perf_test.sh"],
    args = [
        "--target", "//:hybrid_viewer",
        "--duration", "$(perf_duration)",
        "--test-type", "cpu"
    ],
    timeout = "moderate",
    tags = ["performance", "cpu"],
    visibility = ["//visibility:public"],
)

sh_test(
    name = "jolt_physics_perf",
    srcs = ["perf_test.sh"],
    args = [
        "--target", "//modules/physics:jolt_physics",
        "--duration", "$(perf_duration)",
        "--test-type", "physics"
    ],
    data = ["//modules/physics:jolt_physics"],
    timeout = "moderate",
    tags = ["performance", "physics"],
)