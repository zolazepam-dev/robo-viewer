#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <mutex>
#include <map>
#include <atomic>
#include <iostream>
#include <iomanip>

// ============================================================================
// PERFORMANCE DIAGNOSER
// ============================================================================
// Lightweight diagnostic tool to identify "silly simple" stutters
// Tracks mutex contention, loop timing, and periodic stalls
// ============================================================================

class PerformanceDiagnoser {
public:
    struct Timer {
        std::chrono::high_resolution_clock::time_point start;
        std::string name;
        
        Timer(const std::string& n) : name(n) {
            start = std::chrono::high_resolution_clock::now();
        }
        
        ~Timer() {
            auto end = std::chrono::high_resolution_clock::now();
            float duration = std::chrono::duration<float, std::milli>(end - start).count();
            PerformanceDiagnoser::Get().Record(name, duration);
        }
    };

    static PerformanceDiagnoser& Get() {
        static PerformanceDiagnoser instance;
        return instance;
    }

    void Record(const std::string& name, float durationMs) {
        std::lock_guard<std::mutex> lock(mMutex);
        auto& stat = mStats[name];
        stat.totalMs += durationMs;
        stat.count++;
        stat.maxMs = std::max(stat.maxMs, durationMs);
        
        // Detect "Stutter" (anything > 50ms)
        if (durationMs > 50.0f) {
            std::cerr << "[STUTTER DETECTED] " << name << ": " << durationMs << "ms" << std::endl;
        }
    }

    void PrintReport() {
        std::lock_guard<std::mutex> lock(mMutex);
        std::cout << "\n========== PERFORMANCE DIAGNOSTIC REPORT ==========\n";
        std::cout << std::left << std::setw(30) << "Metric" 
                  << std::setw(10) << "Avg (ms)" 
                  << std::setw(10) << "Max (ms)" 
                  << std::setw(10) << "Count" << "\n";
        std::cout << "---------------------------------------------------\n";
        
        for (auto const& [name, stat] : mStats) {
            float avg = stat.totalMs / stat.count;
            std::cout << std::left << std::setw(30) << name 
                      << std::setw(10) << std::fixed << std::setprecision(3) << avg 
                      << std::setw(10) << stat.maxMs 
                      << std::setw(10) << stat.count << "\n";
        }
        std::cout << "===================================================\n" << std::endl;
    }

private:
    struct Stat {
        float totalMs = 0;
        float maxMs = 0;
        long long count = 0;
    };
    
    std::map<std::string, Stat> mStats;
    std::mutex mMutex;
    
    PerformanceDiagnoser() = default;
};

// Macros for easy profiling
#define DIAGNOSE_SCOPE(name) PerformanceDiagnoser::Timer diagnoser_timer(name)
#define DIAGNOSE_MUTEX_LOCK(mutex_var, name) \
    float wait_start_##__LINE__ = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(); \
    std::lock_guard<std::mutex> lock_##__LINE__(mutex_var); \
    float wait_end_##__LINE__ = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(); \
    PerformanceDiagnoser::Get().Record(std::string("Mutex Wait: ") + name, wait_end_##__LINE__ - wait_start_##__LINE__);
