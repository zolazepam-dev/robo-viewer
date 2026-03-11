#!/bin/bash
# Build script for WebAssembly RL Pipeline

# Check if Emscripten is available
if ! command -v em++ &> /dev/null; then
    echo "Emscripten (em++) not found. Please install Emscripten first."
    echo "You can install it via:"
    echo "  curl -L https://github.com/emscripten-core/emsdk/archive/master.tar.gz | tar xz"
    echo "  cd emsdk-master"
    echo "  ./emsdk install latest"
    echo "  ./emsdk activate latest"
    echo "  source emsdk_env.sh"
    exit 1
fi

# Build directory
BUILD_DIR="build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Building WebAssembly RL Pipeline..."

emcmake ..

if [ $? -eq 0 ]; then
    echo "Build configuration successful!"
    echo "Running make..."
    emmake make
    
    if [ $? -eq 0 ]; then
        echo "Build completed successfully!"
        echo "Output: rl_pipeline.wasm"
    else
        echo "Make failed. Check the error messages above."
        exit 1
    fi
else
    echo "CMake configuration failed. Check the error messages above."
    exit 1
fi

echo "Build process completed."
echo "To test the module:"
echo "  cd /media/cammyz/EverythingHere/robo-viewer/wasm_rl_pipeline"
echo "  node test_module.js" # Example test script