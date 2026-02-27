#!/bin/bash

# Exit on any error
set -e

echo "=== 1. Installing System Dependencies (Ubuntu/Debian) ==="
# Note: If you are on macOS, skip this and use: brew install cmake glfw freetype

# Check if we have sudo access
if ! sudo -l &> /dev/null; then
    echo "ERROR: This script requires sudo privileges to install system dependencies"
    echo "Please run this script as a user with sudo access"
    exit 1
fi

sudo apt-get update
sudo apt-get install -y git g++ cmake libglfw3-dev libfreetype6-dev libgl1-mesa-dev

echo -e "\n=== 2. Downloading Morphologica ==="
if [ ! -d "morphologica" ]; then
    git clone https://github.com/ABRG-Models/morphologica.git
else
    echo "Morphologica directory already exists. Skipping clone."
fi

echo -e "\n=== 3. Building and Installing Morphologica ==="
# Morphologica needs to run through CMake to generate version headers
cd morphologica
mkdir -p build
cd build
cmake ..
make -j$(nproc)
sudo make install
cd ../..

echo -e "\n=== 4. Compiling MicroBoard ==="
# Morphologica requires C++20. We link against GLFW, FreeType, and OpenGL.
g++ micro_board.cpp -o micro_board -std=c++20 \
    -I/usr/local/include/freetype2 \
    -lglfw -lGL -lfreetype

echo -e "\n=== Done! ==="
echo "You can now test the board by piping some mock CSV data into it:"
echo -e "1,Loss,1.5\n2,Loss,0.9\n3,Loss,0.4" | ./micro_board