#!/bin/bash
# Dedicated build script for micro_board to bypass Bazel sandbox limitations with morphologica incbin

set -e

echo "Building micro_board with C++20..."

# Detect freetype include path
FREETYPE_INC=$(pkg-config --cflags freetype2)

# Compile micro_board.cpp directly
g++ -std=c++20 -O3 micro_board.cpp \
    -Imorphologica \
    $FREETYPE_INC \
    -DMORPH_FONTS_DIR=\"fonts\" \
    -lGL -lfreetype -lpthread -lglfw -lGLEW \
    -o micro_board_gui

echo "Successfully built micro_board_gui"
