#include "ParallelPhysics.h"
#include <iostream>

ParallelPhysics::ParallelPhysics(const Config& config)
    : mNumSystems(config.numPhysicsSystems),
      mEnvsPerSystem(config.envsPerSystem),
      mTimestep(config.timestep) {
}

ParallelPhysics::~ParallelPhysics() {
    // Signal all workers to stop
    for (auto& batch : mBatches) {
        batch->workReady.store(false);
    }
    
    // Wait for workers to finish
    for (auto& batch : mBatches) {
        if (batch->workerThread.joinable()) {
            batch->workerThread.join();
        }
    }
}

void ParallelPhysics::Init(const std::string& robotConfigPath) {
    mRobotConfigPath = robotConfigPath;
    
    // Initialize each physics batch
    for (int i = 0; i < mNumSystems; ++i) {
        auto batch = std::make_unique<PhysicsBatch>();
        batch->core = std::make_unique<PhysicsCore>();
        batch->core->Init(mEnvsPerSystem);
        batch->workReady.store(false);
        batch->workDone.store(false);
        
        // Assign environment indices to this batch
        for (int j = 0; j < mEnvsPerSystem; ++j) {
            int envIdx = i * mEnvsPerSystem + j;
            batch->envIndices.push_back(envIdx);
        }
        
        // Start worker thread
        PhysicsBatch* batchPtr = batch.get();
        batch->workerThread = std::thread(PhysicsWorkerThread, batchPtr, mTimestep);
        
        mBatches.emplace_back(std::move(batch));
    }
    
    std::cout << "[ParallelPhysics] Initialized " << mNumSystems 
              << " physics systems × " << mEnvsPerSystem << " envs = " 
              << (mNumSystems * mEnvsPerSystem) << " total envs" << std::endl;
}

void ParallelPhysics::PhysicsWorkerThread(PhysicsBatch* batch, float timestep) {
    while (true) {
        // Wait for work signal
        while (!batch->workReady.load()) {
            std::this_thread::yield();
        }
        
        // Step physics for all envs in this batch
        batch->core->GetPhysicsSystem().Update(
            timestep, 1, 
            batch->core->GetTempAllocator(), 
            batch->core->GetJobSystem()
        );
        
        // Signal completion
        batch->workDone.store(true);
        
        // Reset for next cycle
        batch->workReady.store(false);
    }
}

void ParallelPhysics::StepAllParallel() {
    // Signal all workers to start
    for (auto& batch : mBatches) {
        batch->workReady.store(true);
    }
    
    // Wait for all workers to complete
    for (auto& batch : mBatches) {
        while (!batch->workDone.load()) {
            std::this_thread::yield();
        }
        batch->workDone.store(false);
    }
}

PhysicsCore* ParallelPhysics::GetPhysicsCore(int envIndex) {
    int batchIdx = envIndex / mEnvsPerSystem;
    if (batchIdx >= 0 && batchIdx < mNumSystems) {
        return mBatches[batchIdx]->core.get();
    }
    return nullptr;
}

JPH::PhysicsSystem* ParallelPhysics::GetPhysicsSystem(int envIndex) {
    PhysicsCore* core = GetPhysicsCore(envIndex);
    return core ? &core->GetPhysicsSystem() : nullptr;
}
