#!/bin/bash
# Memory-safe build script for low-RAM systems

# Limit Bazel to 4GB RAM and 4 concurrent jobs
export BAZEL_OPTS="--jobs=4 --local_ram_resources=4096"

# Reduce Bazel's internal memory pressure
export BAZEL_STARTUP_OPTS="--max_idle_secs=60"

echo "🔧 Building with memory limits:"
echo "   - Max jobs: 4"
echo "   - RAM limit: 4GB"
echo "   - Available RAM: $(free -h | awk '/^Mem:/{print $7}')"
echo ""

# Clean previous builds to free memory
bazel clean --expunge 2>/dev/null

# Build with memory constraints
bazel build $BAZEL_OPTS "$@"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "✅ Build successful!"
else
    echo "❌ Build failed with exit code: $exit_code"
    echo "💡 Try: bazel build --jobs=2 --local_ram_resources=2048 $@"
fi

exit $exit_code
