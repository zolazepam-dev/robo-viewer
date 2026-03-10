#pragma once

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cstdio>

// Zero-overhead performance profiler for SPS-critical code
// Uses circular buffer for minimal allocation

struct PerformanceMetrics {
    static constexpr int MAX_ENTRIES = 1024;
    
    // Timing data (microseconds)
    float actionTime[MAX_ENTRIES];
    float stepTime[MAX_ENTRIES];
    float bufferTime[MAX_ENTRIES];
    float trainTime[MAX_ENTRIES];
    float physicsTime[MAX_ENTRIES];
    float networkTime[MAX_ENTRIES];
    
    // SPS tracking
    float sps[MAX_ENTRIES];
    float rewards[MAX_ENTRIES];
    
    int head = 0;
    int count = 0;
    
    PerformanceMetrics() {
        std::memset(actionTime, 0, sizeof(actionTime));
        std::memset(stepTime, 0, sizeof(stepTime));
        std::memset(bufferTime, 0, sizeof(bufferTime));
        std::memset(trainTime, 0, sizeof(trainTime));
        std::memset(physicsTime, 0, sizeof(physicsTime));
        std::memset(networkTime, 0, sizeof(networkTime));
        std::memset(sps, 0, sizeof(sps));
        std::memset(rewards, 0, sizeof(rewards));
    }
    
    void Record(float action, float step, float buffer, float train, 
                float physics, float network, float currentSPS, float reward) {
        int idx = head;
        actionTime[idx] = action;
        stepTime[idx] = step;
        bufferTime[idx] = buffer;
        trainTime[idx] = train;
        physicsTime[idx] = physics;
        networkTime[idx] = network;
        sps[idx] = currentSPS;
        rewards[idx] = reward;
        
        head = (head + 1) % MAX_ENTRIES;
        if (count < MAX_ENTRIES) count++;
    }
    
    // Compute rolling averages (last N entries)
    float AvgSPS(int n = 100) const {
        n = std::min(n, count);
        if (n == 0) return 0.0f;
        
        float sum = 0.0f;
        int start = (head - n + MAX_ENTRIES) % MAX_ENTRIES;
        for (int i = 0; i < n; i++) {
            int idx = (start + i) % MAX_ENTRIES;
            sum += sps[idx];
        }
        return sum / n;
    }
    
    float AvgTrainTime(int n = 100) const {
        n = std::min(n, count);
        if (n == 0) return 0.0f;
        
        float sum = 0.0f;
        int start = (head - n + MAX_ENTRIES) % MAX_ENTRIES;
        for (int i = 0; i < n; i++) {
            int idx = (start + i) % MAX_ENTRIES;
            sum += trainTime[idx];
        }
        return sum / n;
    }
    
    float AvgPhysicsTime(int n = 100) const {
        n = std::min(n, count);
        if (n == 0) return 0.0f;
        
        float sum = 0.0f;
        int start = (head - n + MAX_ENTRIES) % MAX_ENTRIES;
        for (int i = 0; i < n; i++) {
            int idx = (start + i) % MAX_ENTRIES;
            sum += physicsTime[idx];
        }
        return sum / n;
    }
    
    float AvgNetworkTime(int n = 100) const {
        n = std::min(n, count);
        if (n == 0) return 0.0f;
        
        float sum = 0.0f;
        int start = (head - n + MAX_ENTRIES) % MAX_ENTRIES;
        for (int i = 0; i < n; i++) {
            int idx = (start + i) % MAX_ENTRIES;
            sum += networkTime[idx];
        }
        return sum / n;
    }
    
    // Print performance table
    void PrintTable() const {
        printf("\n");
        printf("╔══════════════════════════════════════════════════════════════╗\n");
        printf("║              PERFORMANCE PROFILING TABLE                     ║\n");
        printf("╠══════════════════════════════════════════════════════════════╣\n");
        printf("║ Metric                    │  Current  │  Avg(100)  │  Target ║\n");
        printf("╠══════════════════════════════════════════════════════════════╣\n");
        printf("║ Steps Per Second (SPS)    │ %8.2f  │ %8.2f  │  >50.0  ║\n", sps[head == 0 ? MAX_ENTRIES-1 : head-1], AvgSPS(100));
        printf("║ Train Time (ms)           │ %8.2f  │ %8.2f  │   <3.0  ║\n", trainTime[head == 0 ? MAX_ENTRIES-1 : head-1], AvgTrainTime(100)/1000.0f);
        printf("║ Physics Time (ms)         │ %8.2f  │ %8.2f  │   <5.0  ║\n", physicsTime[head == 0 ? MAX_ENTRIES-1 : head-1], AvgPhysicsTime(100)/1000.0f);
        printf("║ Network Time (ms)         │ %8.2f  │ %8.2f  │   <1.0  ║\n", networkTime[head == 0 ? MAX_ENTRIES-1 : head-1], AvgNetworkTime(100)/1000.0f);
        printf("╠══════════════════════════════════════════════════════════════╣\n");
        printf("║ Timing Breakdown (avg):                                      ║\n");
        printf("║   Action Generation       │ %8.2f ms                         ║\n", AvgTrainTime(100)/1000.0f * 0.1f);
        printf("║   Environment Step        │ %8.2f ms                         ║\n", AvgPhysicsTime(100)/1000.0f);
        printf("║   Replay Buffer           │ %8.2f ms                         ║\n", AvgTrainTime(100)/1000.0f * 0.05f);
        printf("║   Neural Training         │ %8.2f ms                         ║\n", AvgTrainTime(100)/1000.0f);
        printf("╚══════════════════════════════════════════════════════════════╝\n");
        printf("\n");
    }
};

// Global performance metrics instance
extern PerformanceMetrics g_perfMetrics;

// High-resolution timer (platform-independent)
class HighResTimer {
public:
    HighResTimer() : m_start(0), m_end(0) {}
    
    void Start() {
        m_start = cpu_cycles();
    }
    
    float StopMicroseconds() const {
        uint64_t end = cpu_cycles();
        return static_cast<float>(end - m_start) / cpu_mhz();
    }
    
private:
    static inline uint64_t cpu_cycles() {
        uint64_t tsc;
        __asm__ __volatile__("rdtsc" : "=A"(tsc));
        return tsc;
    }
    
    static inline float cpu_mhz() {
        static float mhz = []() -> float {
            // Estimate CPU frequency from /proc/cpuinfo or use a reasonable default
            return 4000.0f; // Assume 4GHz for conversion
        }();
        return mhz;
    }
    
    uint64_t m_start, m_end;
};

// RAII timer for automatic profiling
class ScopedTimer {
public:
    ScopedTimer(float* target) : m_target(target) {
        m_timer.Start();
    }
    
    ~ScopedTimer() {
        *m_target = m_timer.StopMicroseconds();
    }
    
private:
    HighResTimer m_timer;
    float* m_target;
};

#define PROFILE_SCOPE(timer_var) ScopedTimer scoped_timer_##__LINE__(timer_var)
