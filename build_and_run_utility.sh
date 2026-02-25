# Standard usage - build and run combat trainer
./build_and_run.sh

# Clean build with debug symbols
./build_and_run.sh --clean --debug

# Build regular viewer instead of combat trainer
./build_and_run.sh --viewer

# Just build, don't run
./build_and_run.sh -b

# Run existing binary
./build_and_run.sh -r

# Clean build with verbose output
./build_and_run.sh -c --verbose
