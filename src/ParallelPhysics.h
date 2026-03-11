#pragma once

#include "PhysicsCore.h"
#include <vector>
#include <thread>
#include <atomic>
#include <memory>

// Multi-Physics System for parallel environment stepping
// Splits N environments across M physics systems for parallel execution
class ParallelPhysics {
public:
    struct Config {
        int numPhysicsSystems;
        int envsPerSystem;
        float timestep;
        
        Config() : numPhysicsSystems(8), envsPerSystem(32), timestep(1.0f / 120.0f) {}
    };
    
    ParallelPhysics(const Config& config = Config());
    ~ParallelPhysics();
    
    // Initialize all physics systems
    void Init(const std::string& robotConfigPath);
    
    // Step all physics systems in parallel
    void StepAllParallel();
    
    // Get physics system for environment i
    PhysicsCore* GetPhysicsCore(int envIndex);
    JPH::PhysicsSystem* GetPhysicsSystem(int envIndex);
    
    // Get number of physics systems
    int GetNumSystems() const { return mNumSystems; }
    
    // Get environments per system
    int GetEnvsPerSystem() const { return mEnvsPerSystem; }

private:
    struct PhysicsBatch {
        PhysicsCore* core;  // Pointer to avoid move issues
        std::vector<int> envIndices;  // Which envs use this physics system
        std::thread workerThread;
        std::atomic<bool> workReady;
        std::atomic<bool> workDone;
        
        PhysicsBatch() : core(nullptr), workReady(false), workDone(false) {}
        ~PhysicsBatch() { delete core; }
        PhysicsBatch(const PhysicsBatch&) = delete;
        PhysicsBatch& operator=(const PhysicsBatch&) = delete;
    };
    
    std::vector<std::unique_ptr<PhysicsBatch>> mBatches;
    int mNumSystems;
    int mEnvsPerSystem;
    float mTimestep;
    std::string mRobotConfigPath;
    
    // Worker thread function for parallel stepping
    static void PhysicsWorkerThread(PhysicsBatch* batch, float timestep);
};
