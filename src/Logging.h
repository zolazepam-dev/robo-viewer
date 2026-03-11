#pragma once

#include <cstdio>
#include <cstdint>
#include <chrono>

/**
 * Logging Utility for JOLTrl
 * 
 * Usage:
 *   LOG_INFO("Message: %d", value);
 *   LOG_DEBUG("Debug: %f", debugValue);
 *   LOG_WARN("Warning: %s", msg);
 *   LOG_ERROR("Error: %s", errorMsg);
 * 
 * To disable all logging in release builds, define NO_LOGGING:
 *   - Add -DNO_LOGGING to compiler flags
 *   - Or #define NO_LOGGING before including this header
 */

// Logging levels (must be macros for preprocessor checks)
#define JOLTRL_LOG_LEVEL_NONE  0
#define JOLTRL_LOG_LEVEL_ERROR 1
#define JOLTRL_LOG_LEVEL_WARN  2
#define JOLTRL_LOG_LEVEL_INFO  3
#define JOLTRL_LOG_LEVEL_DEBUG 4

// Default log level (can be overridden)
#ifndef JOLTRL_LOG_LEVEL
    #ifdef NO_LOGGING
        #define JOLTRL_LOG_LEVEL JOLTRL_LOG_LEVEL_NONE
    #else
        #define JOLTRL_LOG_LEVEL JOLTRL_LOG_LEVEL_INFO
    #endif
#endif

// Internal logging macro (do not use directly)
#define JOLTRL_LOG_INTERNAL(level_name, level_val, format, ...) \
    do { \
        if (level_val <= JOLTRL_LOG_LEVEL) { \
            FILE* stream = (level_val <= JOLTRL_LOG_LEVEL_WARN) ? stderr : stdout; \
            std::fprintf(stream, "[%s] " format "\n", level_name, ##__VA_ARGS__); \
            std::fflush(stream); \
        } \
    } while(0)

// Public logging macros
#if JOLTRL_LOG_LEVEL >= JOLTRL_LOG_LEVEL_ERROR
    #define LOG_ERROR(format, ...) JOLTRL_LOG_INTERNAL("ERROR", JOLTRL_LOG_LEVEL_ERROR, format, ##__VA_ARGS__)
#else
    #define LOG_ERROR(format, ...) ((void)0)
#endif

#if JOLTRL_LOG_LEVEL >= JOLTRL_LOG_LEVEL_WARN
    #define LOG_WARN(format, ...) JOLTRL_LOG_INTERNAL("WARN", JOLTRL_LOG_LEVEL_WARN, format, ##__VA_ARGS__)
#else
    #define LOG_WARN(format, ...) ((void)0)
#endif

#if JOLTRL_LOG_LEVEL >= JOLTRL_LOG_LEVEL_INFO
    #define LOG_INFO(format, ...) JOLTRL_LOG_INTERNAL("INFO", JOLTRL_LOG_LEVEL_INFO, format, ##__VA_ARGS__)
#else
    #define LOG_INFO(format, ...) ((void)0)
#endif

#if JOLTRL_LOG_LEVEL >= JOLTRL_LOG_LEVEL_DEBUG
    #define LOG_DEBUG(format, ...) JOLTRL_LOG_INTERNAL("DEBUG", JOLTRL_LOG_LEVEL_DEBUG, format, ##__VA_ARGS__)
#else
    #define LOG_DEBUG(format, ...) ((void)0)
#endif

// Performance-critical logging (completely disabled in NO_LOGGING builds)
#ifdef NO_LOGGING
    #define LOG_PERF(format, ...) ((void)0)
    #define LOG_TRAIN(format, ...) ((void)0)
    #define LOG_SIM(format, ...) ((void)0)
#else
    #define LOG_PERF(format, ...) LOG_INFO(format, ##__VA_ARGS__)
    #define LOG_TRAIN(format, ...) LOG_DEBUG(format, ##__VA_ARGS__)
    #define LOG_SIM(format, ...) LOG_DEBUG(format, ##__VA_ARGS__)
#endif

// Initialization function - call once at startup
inline void InitLogging(int level = JOLTRL_LOG_LEVEL) {
    #ifndef NO_LOGGING
        fprintf(stdout, "[Logging] Initialized with log level: %d\n", level);
        fprintf(stdout, "[Logging] NO_LOGGING is NOT defined (logging enabled)\n");
    #else
        fprintf(stdout, "[Logging] NO_LOGGING is defined (logging disabled for performance)\n");
    #endif
    fflush(stdout);
}

// Performance counter utility
class PerformanceCounter {
public:
    inline PerformanceCounter(const char* name) 
        : mName(name), mStart(std::chrono::high_resolution_clock::now()) {}
    
    inline ~PerformanceCounter() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - mStart).count();
        #ifndef NO_LOGGING
            if (JOLTRL_LOG_LEVEL >= JOLTRL_LOG_LEVEL_DEBUG) {
                fprintf(stdout, "[PERF] %s: %ld µs\n", mName, (long)duration);
                fflush(stdout);
            }
        #endif
    }
    
private:
    const char* mName;
    std::chrono::high_resolution_clock::time_point mStart;
};

// Convenience macro for performance timing
#ifndef NO_LOGGING
    #define PERF_SCOPE(name) PerformanceCounter perf_counter_##__LINE__(name)
#else
    #define PERF_SCOPE(name) ((void)0)
#endif
