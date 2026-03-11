cmake_minimum_required(VERSION 3.20)
project(wasm_rl_pipeline)

# Set Emscripten as the compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(EMSCRIPTEN)
    message(STATUS "Building with Emscripten for WebAssembly")
else()
    message(FATAL_ERROR "This project must be built with Emscripten. Use: emcmake cmake ..")
endif()

# Jolt Physics setup
set(JOLT_PHYSICS_ROOT "/media/cammyz/EverythingHere/robo-viewer/third_party/Jolt")
set(JOLT_PHYSICS_INCLUDE_DIR "${JOLT_PHYSICS_ROOT}")
set(JOLT_PHYSICS_LIB "${JOLT_PHYSICS_ROOT}/build_wasm/libJolt.a")

# Include directories
include_directories(
    ${JOLT_PHYSICS_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/wasm_module
)

# Source files for the WebAssembly module
set(WASM_SOURCES
    wasm_module/jolt_physics_bindings.cpp
    wasm_module/rl_pipeline.cpp
    wasm_module/vectorized_env.cpp
    wasm_module/combat_env.cpp
    wasm_module/training_coordinator.cpp
    wasm_module/webworker_manager.cpp
)

# Build the WebAssembly module
add_executable(rl_pipeline.wasm.js ${WASM_SOURCES})

# Link Jolt Physics
target_link_libraries(rl_pipeline.wasm.js PRIVATE ${JOLT_PHYSICS_LIB})

# Emscripten specific settings
set_target_properties(rl_pipeline.wasm.js PROPERTIES
    LINK_FLAGS "
        -s WASM=1
        -s MODULARIZE=1
        -s EXPORT_NAME=\"RLPipelineModule\"
        -s ALLOW_MEMORY_GROWTH=1
        -s MAXIMUM_MEMORY=4GB
        -s STACK_SIZE=8MB
        -s PTHREAD_POOL_SIZE=8
        -s USE_PTHREADS=1
        -s ENVIRONMENT=web,worker
        -lidbfs.js
        -s EXPORTED_FUNCTIONS=['_malloc','_free','_main']
        -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap','getValue','setValue','HEAP8','HEAP32','HEAPF32','HEAPF64']
    "
)

# Additional compiler flags
target_compile_options(rl_pipeline.wasm.js PRIVATE
    -O3
    -flto
    -fno-unwind-tables
    -fno-asynchronous-unwind-tables
    -DEMSCRIPTEN_HAS_UNBOUND_TYPE_NAMES=0
)