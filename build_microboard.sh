#!/bin/bash

# Build microboard directly with g++ (bypassing bazel font issues)

echo "=== Building MicroBoard with Morphologica ==="

cd /home/cammyz/robo-viewer

# Get morphologica include path
MORPH_PATH="/home/cammyz/robo-viewer/morphologica"
FONT_PATH="/home/cammyz/robo-viewer/morphologica/fonts"

g++ micro_board.cpp -o micro_board_gui -std=c++20 \
    -I${MORPH_PATH} \
    -I/usr/include/freetype2 \
    -DMORPH_FONTS_DIR=\"${FONT_PATH}\" \
    -lGL -lglfw -lfreetype -lpthread \
    -O3

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Run with: ./micro_board_gui or pipe CSV data to it"
else
    echo "Build failed"
    exit 1
fi