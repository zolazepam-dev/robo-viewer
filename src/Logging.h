#pragma once

#include <cstdio>
#include <cstdint>

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

// Logging levels
enum class LogLevel {
    NONE = 0,
    ERROR = 1,
    WARN = 2,
    INFO = 3,
    DEBUG = 4
};

// Default log level (can be overridden)
#ifndef JOLTRL_LOG_LEVEL
    #ifdef NO_LOGGING
        #define JOLTRL_LOG_LEVEL LogLevel::NONE
    #else
        #define JOLTRL_LOG_LEVEL LogLevel::INFO
    #endif
#endif

// Internal logging macro (do not use directly)
#define JOLTRL_LOG_INTERNAL(level, format, ...) \
    do { \
        if (static_cast<int>(level) <= static_cast<int>(JOLTRL_LOG_LEVEL)) { \
            std::fprintf(stderr, "[%s] " format "\n", #level, ##__VA_ARGS__); \
            std::fflush(stderr); \
        } \
    } while(0)

// Public logging macros
#if JOLTRL_LOG_LEVEL >= LogLevel::ERROR
    #define LOG_ERROR(format, ...) JOLTRL_LOG_INTERNAL(ERROR, format, ##__VA_ARGS__)
#else
    #define LOG_ERROR(format, ...) ((void)0)
#endif

#if JOLTRL_LOG_LEVEL >= LogLevel::WARN
    #define LOG_WARN(format, ...) JOLTRL_LOG_INTERNAL(WARN, format, ##__VA_ARGS__)
#else
    #define LOG_WARN(format, ...) ((void)0)
#endif

#if JOLTRL_LOG_LEVEL >= LogLevel::INFO
    #define LOG_INFO(format, ...) JOLTRL_LOG_INTERNAL(INFO, format, ##__VA_ARGS__)
#else
    #define LOG_INFO(format, ...) ((void)0)
#endif

#if JOLTRL_LOG_LEVEL >= LogLevel::DEBUG
    #define LOG_DEBUG(format, ...) JOLTRL_LOG_INTERNAL(DEBUG, format, ##__VA_ARGS__)
#else
    #define LOG_DEBUG(format, ...) ((void)0)
#endif

// Performance-critical logging (completely disabled in NO_LOGGING builds)
#ifdef NO_LOGGING
    #define LOG_PERF(format, ...) ((void)0)
    #define LOG_TRAIN(format, ...) ((void)0)
    #define LOG_SIM(format, ...) ((void)0)
#else
    #define LOG_PERF(format, ...) JOLTRL_LOG_INTERNAL(INFO, format, ##__VA_ARGS__)
    #define LOG_TRAIN(format, ...) JOLTRL_LOG_INTERNAL(DEBUG, format, ##__VA_ARGS__)
    #define LOG_SIM(format, ...) JOLTRL_LOG_INTERNAL(DEBUG, format, ##__VA_ARGS__)
#endif

// Initialization function - call once at startup
inline void InitLogging(LogLevel level = JOLTRL_LOG_LEVEL) {
    #ifndef NO_LOGGING
        fprintf(stderr, "[Logging] Initialized with log level: %d\n", static_cast<int>(level));
        fprintf(stderr, "[Logging] NO_LOGGING is NOT defined (logging enabled)\n");
    #else
        fprintf(stderr, "[Logging] NO_LOGGING is defined (logging disabled for performance)\n");
    #endif
    fflush(stderr);
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
            if (static_cast<int>(JOLTRL_LOG_LEVEL) >= static_cast<int>(LogLevel::DEBUG)) {
                fprintf(stderr, "[PERF] %s: %lld µs\n", mName, duration);
                fflush(stderr);
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
