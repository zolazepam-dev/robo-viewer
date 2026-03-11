cmake_minimum_required(VERSION 3.16)
project(MultiAgentRL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optional: Eigen for linear algebra (header-only)
find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
    add_definitions(-DUSE_EIGEN)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# Main library sources (header-only, stubs for build system)
set(LIB_SOURCES
    src/environment.cpp
    src/neural_net.cpp
    src/rl_algo.cpp
    src/parallel.cpp
)

# Main library
add_library(marl_lib STATIC ${LIB_SOURCES})
target_include_directories(marl_lib PUBLIC include)

# Main executable
add_executable(marl_train src/main.cpp)
target_link_libraries(marl_train marl_lib pthread)

# Test executables
add_executable(test_env tests/test_environment.cpp)
target_link_libraries(test_env marl_lib)

add_executable(test_nn tests/test_neural_net.cpp)
target_link_libraries(test_nn marl_lib)

add_executable(test_rl tests/test_rl_algo.cpp)
target_link_libraries(test_rl marl_lib)

add_executable(test_parallel tests/test_parallel.cpp)
target_link_libraries(test_parallel marl_lib pthread)

add_executable(test_integration tests/test_integration.cpp)
target_link_libraries(test_integration marl_lib pthread)

# Enable testing
enable_testing()

# Custom test target that runs all tests
add_custom_target(run_tests
    COMMAND echo "Running Environment Tests..." && ./test_env
    COMMAND echo "Running Neural Net Tests..." && ./test_nn
    COMMAND echo "Running RL Algo Tests..." && ./test_rl
    COMMAND echo "Running Parallel Tests..." && ./test_parallel
    COMMAND echo "Running Integration Tests..." && ./test_integration
    COMMAND echo "All tests complete!"
    DEPENDS test_env test_nn test_rl test_parallel test_integration
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
