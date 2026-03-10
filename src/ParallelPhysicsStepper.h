#pragma once

#include "PhysicsCore.h"
#include "AlignedAllocator.h"
#include "OptimizedBatchOps.h"
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>

// ============================================================================
// PARALLEL PHYSICS STEPPER
// ============================================================================
// Splits N environments across M physics systems for parallel execution
// Each physics system runs on its own thread, stepping independently
// 
// Architecture:
// - Main thread: Collects actions, harvests states
// - Worker threads: Step physics systems in parallel
// - Lock-free communication via double-buffered action/state queues
// ============================================================================

class ParallelPhysicsStepper {
public:
    struct Config {
        int numPhysicsSystems;      // Number of parallel physics systems (typically = num CPU cores)
        int envsPerSystem;          // Environments per physics system
        float timestep;             // Physics timestep
        int substeps;               // Physics substeps per frame
        
        Config() 
            : numPhysicsSystems(8), envsPerSystem(32), timestep(1.0f / 120.0f), substeps(1) {}
    };
    
    ParallelPhysicsStepper(const Config& config = Config());
    ~ParallelPhysicsStepper();
    
    // Initialize all physics systems with robot configurations
    void Init(const std::string& robotConfigPath);
    
    // Shutdown all physics systems and worker threads
    void Shutdown();
    
    // Queue actions for all environments (called from main thread)
    // actions: [numSystems * envsPerSystem * actionDim]
    void QueueActions(const float* actions, int actionDim);
    
    // Step all physics systems in parallel (non-blocking)
    void StepAllParallel();
    
    // Wait for all physics steps to complete
    void WaitForStepComplete();
    
    // Harvest states from all environments (called after step complete)
    // Returns pointers to internal buffers (zero-copy)
    void HarvestStates(float* observations, float* rewards, bool* dones, int obsDim);
    
    // Get physics core for environment i
    PhysicsCore* GetPhysicsCore(int envIndex);
    JPH::PhysicsSystem* GetPhysicsSystem(int envIndex);
    
    // Get number of physics systems
    int GetNumSystems() const { return mNumSystems; }
    
    // Get environments per system
    int GetEnvsPerSystem() const { return mEnvsPerSystem; }
    
    // Get total environments
    int GetTotalEnvs() const { return mNumSystems * mEnvsPerSystem; }
    
    // Check if stepping is complete (non-blocking)
    bool IsStepComplete() const;

private:
    struct alignas(64) PhysicsBatch {
        PhysicsCore* core;                      // Physics system for this batch
        std::vector<int> envIndices;            // Which envs use this physics system
        std::thread workerThread;               // Worker thread
        
        // Double-buffered action queues
        AlignedVector32<float> currentActions;  // Actions being consumed
        AlignedVector32<float> nextActions;     // Actions being produced
        std::atomic<int> activeBuffer;          // 0 = current, 1 = next
        
        // Synchronization
        std::atomic<bool> workReady;            // Work available
        std::atomic<bool> workDone;             // Work completed
        std::atomic<bool> shouldStop;           // Shutdown signal
        
        // Per-batch state buffers (SoA layout)
        AlignedVector32<float> observations;    // [envs * obsDim]
        AlignedVector32<float> rewards;         // [envs]
        std::vector<bool> dones;                // [envs]
        
        char padding[64 - (sizeof(std::atomic<bool>) % 64)];  // Cache line padding
        
        PhysicsBatch() 
            : core(nullptr), activeBuffer(0), workReady(false), workDone(false), shouldStop(false) {}
        
        ~PhysicsBatch() { 
            if (workerThread.joinable()) {
                workerThread.join();
            }
            delete core; 
        }
        
        // Prevent copying (move only)
        PhysicsBatch(const PhysicsBatch&) = delete;
        PhysicsBatch& operator=(const PhysicsBatch&) = delete;
    };
    
    std::vector<std::unique_ptr<PhysicsBatch>> mBatches;
    int mNumSystems;
    int mEnvsPerSystem;
    float mTimestep;
    int mSubsteps;
    std::string mRobotConfigPath;
    
    // Global state buffers (for harvest)
    AlignedVector32<float> mGlobalObservations;
    AlignedVector32<float> mGlobalRewards;
    std::vector<bool> mGlobalDones;
    
    // Worker thread function
    static void PhysicsWorkerThread(PhysicsBatch* batch, float timestep, int substeps);
    
    // Environment to physics system mapping
    int EnvToSystemIndex(int envIndex) const { return envIndex / mEnvsPerSystem; }
    int EnvToLocalIndex(int envIndex) const { return envIndex % mEnvsPerSystem; }
};

// ============================================================================
// LOCK-FREE STATE HARVESTER
// ============================================================================
// Zero-copy state harvesting from parallel physics systems
// Uses per-thread buffers to avoid contention

class LockFreeStateHarvester {
public:
    struct alignas(64) ThreadLocalBuffer {
        AlignedVector32<float> observations;
        AlignedVector32<float> rewards;
        std::vector<bool> dones;
        std::atomic<bool> hasData;
        char padding[64 - sizeof(std::atomic<bool>) % 64];
        
        ThreadLocalBuffer() : hasData(false) {}
        
        // Explicit move constructor/assignment for atomic member
        ThreadLocalBuffer(ThreadLocalBuffer&& other) noexcept 
            : hasData(other.hasData.load()) {
            observations = std::move(other.observations);
            rewards = std::move(other.rewards);
            dones = std::move(other.dones);
        }
        
        ThreadLocalBuffer& operator=(ThreadLocalBuffer&& other) noexcept {
            if (this != &other) {
                hasData.store(other.hasData.load());
                observations = std::move(other.observations);
                rewards = std::move(other.rewards);
                dones = std::move(other.dones);
            }
            return *this;
        }
        
        // Delete copy operations
        ThreadLocalBuffer(const ThreadLocalBuffer&) = delete;
        ThreadLocalBuffer& operator=(const ThreadLocalBuffer&) = delete;
    };
    
    LockFreeStateHarvester(int numThreads, int obsDim, int envsPerThread);
    ~LockFreeStateHarvester() = default;
    
    // Store harvested state from thread (zero-copy if possible)
    void StoreFromThread(int threadId, const float* observations, const float* rewards, const bool* dones, int numEnvs);
    
    // Merge all thread buffers into single output (contiguous)
    void MergeAll(float* outObservations, float* outRewards, bool* outDones, int obsDim);
    
    // Reset for next step
    void Reset();
    
    // Get buffer for thread (for direct write)
    ThreadLocalBuffer& GetThreadBuffer(int threadId) { return mThreadBuffers[threadId]; }

private:
    std::vector<ThreadLocalBuffer> mThreadBuffers;
    int mNumThreads;
    int mObsDim;
    int mEnvsPerThread;
};

// ============================================================================
// ASYNC ACTION QUEUER
// ============================================================================
// Double-buffered action queue for lock-free producer-consumer pattern
// Main thread produces actions, physics threads consume

class AsyncActionQueuer {
public:
    struct alignas(64) DoubleBuffer {
        AlignedVector32<float> buffer0;
        AlignedVector32<float> buffer1;
        std::atomic<int> readBuffer;   // Which buffer consumers are reading
        std::atomic<int> writeBuffer;  // Which buffer producer is writing
        std::atomic<bool> swapReady;   // Swap is ready
        char padding[64 - sizeof(std::atomic<bool>) % 64];
        
        DoubleBuffer() : readBuffer(0), writeBuffer(0), swapReady(false) {}
    };
    
    AsyncActionQueuer(int totalEnvs, int actionDim);
    ~AsyncActionQueuer() = default;
    
    // Producer: Write next actions (main thread)
    void WriteNextActions(const float* actions, int numActions);
    
    // Consumer: Get current actions (physics thread)
    const float* GetCurrentActions(int& numActions) const;
    
    // Signal that swap is ready (called after all consumers finished)
    void SignalSwapReady();
    
    // Wait for swap to complete (physics thread)
    void WaitForSwap();

private:
    DoubleBuffer mBuffers;
    int mTotalEnvs;
    int mActionDim;
    std::atomic<bool> mSwapComplete;
};

// ============================================================================
// IMPLEMENTATION
// ============================================================================

inline ParallelPhysicsStepper::ParallelPhysicsStepper(const Config& config)
    : mNumSystems(config.numPhysicsSystems)
    , mEnvsPerSystem(config.envsPerSystem)
    , mTimestep(config.timestep)
    , mSubsteps(config.substeps) {
    
    mBatches.resize(mNumSystems);
    for (int i = 0; i < mNumSystems; i++) {
        mBatches[i] = std::make_unique<PhysicsBatch>();
        mBatches[i]->envIndices.reserve(mEnvsPerSystem);
        for (int j = 0; j < mEnvsPerSystem; j++) {
            mBatches[i]->envIndices.push_back(i * mEnvsPerSystem + j);
        }
    }
}

inline ParallelPhysicsStepper::~ParallelPhysicsStepper() {
    Shutdown();
}

inline void ParallelPhysicsStepper::Init(const std::string& robotConfigPath) {
    mRobotConfigPath = robotConfigPath;
    
    // Initialize each physics system
    #pragma omp parallel for
    for (int i = 0; i < mNumSystems; i++) {
        mBatches[i]->core = new PhysicsCore();
        mBatches[i]->core->Init(robotConfigPath);
        
        // Allocate buffers
        int obsDim = 256;  // TODO: Get from config
        int bufferSize = mEnvsPerSystem * obsDim * 2;  // 2 robots per env
        mBatches[i]->observations.resize(bufferSize);
        mBatches[i]->rewards.resize(mEnvsPerSystem * 2);
        mBatches[i]->dones.resize(mEnvsPerSystem * 2);
        
        int actionDim = 32;  // TODO: Get from config
        mBatches[i]->currentActions.resize(mEnvsPerSystem * 2 * actionDim);
        mBatches[i]->nextActions.resize(mEnvsPerSystem * 2 * actionDim);
    }
    
    // Allocate global buffers
    int totalEnvs = mNumSystems * mEnvsPerSystem;
    mGlobalObservations.resize(totalEnvs * 256 * 2);
    mGlobalRewards.resize(totalEnvs * 2);
    mGlobalDones.resize(totalEnvs * 2);
    
    // Start worker threads
    for (int i = 0; i < mNumSystems; i++) {
        mBatches[i]->workerThread = std::thread(PhysicsWorkerThread, 
                                                 mBatches[i].get(), 
                                                 mTimestep, 
                                                 mSubsteps);
    }
}

inline void ParallelPhysicsStepper::Shutdown() {
    // Signal all threads to stop
    for (int i = 0; i < mNumSystems; i++) {
        mBatches[i]->shouldStop.store(true);
        mBatches[i]->workReady.store(true);  // Wake up thread
    }
    
    // Wait for all threads to finish
    for (int i = 0; i < mNumSystems; i++) {
        if (mBatches[i]->workerThread.joinable()) {
            mBatches[i]->workerThread.join();
        }
    }
}

inline void ParallelPhysicsStepper::QueueActions(const float* actions, int actionDim) {
    // Distribute actions to each batch's next buffer
    #pragma omp parallel for
    for (int i = 0; i < mNumSystems; i++) {
        int srcOffset = i * mEnvsPerSystem * 2 * actionDim;
        std::memcpy(mBatches[i]->nextActions.data(), 
                    actions + srcOffset, 
                    mEnvsPerSystem * 2 * actionDim * sizeof(float));
    }
}

inline void ParallelPhysicsStepper::StepAllParallel() {
    // Swap buffers (atomic)
    for (int i = 0; i < mNumSystems; i++) {
        mBatches[i]->activeBuffer.store(1 - mBatches[i]->activeBuffer.load());
        mBatches[i]->workDone.store(false);
        mBatches[i]->workReady.store(true);  // Signal work available
    }
}

inline void ParallelPhysicsStepper::WaitForStepComplete() {
    // Wait for all batches to finish
    for (int i = 0; i < mNumSystems; i++) {
        while (!mBatches[i]->workDone.load()) {
            _mm_pause();  // Hint to CPU that we're spinning
        }
    }
}

inline bool ParallelPhysicsStepper::IsStepComplete() const {
    for (int i = 0; i < mNumSystems; i++) {
        if (!mBatches[i]->workDone.load()) {
            return false;
        }
    }
    return true;
}

inline void ParallelPhysicsStepper::HarvestStates(float* observations, float* rewards, 
                                                   bool* dones, int obsDim) {
    // Merge all batch buffers into global output
    #pragma omp parallel for
    for (int i = 0; i < mNumSystems; i++) {
        int dstOffset = i * mEnvsPerSystem * 2 * obsDim;
        std::memcpy(observations + dstOffset, 
                    mBatches[i]->observations.data(), 
                    mEnvsPerSystem * 2 * obsDim * sizeof(float));
        
        int rewardOffset = i * mEnvsPerSystem * 2;
        std::memcpy(rewards + rewardOffset, 
                    mBatches[i]->rewards.data(), 
                    mEnvsPerSystem * 2 * sizeof(float));
        
        int doneOffset = i * mEnvsPerSystem * 2;
        std::memcpy(dones + doneOffset, 
                    mBatches[i]->dones.data(), 
                    mEnvsPerSystem * 2 * sizeof(bool));
    }
}

inline PhysicsCore* ParallelPhysicsStepper::GetPhysicsCore(int envIndex) {
    int systemIdx = EnvToSystemIndex(envIndex);
    return mBatches[systemIdx]->core;
}

inline JPH::PhysicsSystem* ParallelPhysicsStepper::GetPhysicsSystem(int envIndex) {
    int systemIdx = EnvToSystemIndex(envIndex);
    return &mBatches[systemIdx]->core->GetPhysicsSystem();
}

inline void ParallelPhysicsStepper::PhysicsWorkerThread(PhysicsBatch* batch, 
                                                         float timestep, 
                                                         int substeps) {
    while (!batch->shouldStop.load()) {
        // Wait for work
        while (!batch->workReady.load() && !batch->shouldStop.load()) {
            _mm_pause();
        }
        
        if (batch->shouldStop.load()) break;
        
        // Get active buffer index
        int activeIdx = batch->activeBuffer.load();
        float* actions = (activeIdx == 0) ? batch->currentActions.data() 
                                           : batch->nextActions.data();
        
        // Step physics for all envs in this batch
        // (Implementation depends on PhysicsCore API)
        batch->core->StepMultiple(actions, timestep, substeps, 
                                   batch->observations.data(),
                                   batch->rewards.data(),
                                   batch->dones.data());
        
        // Mark work as done
        batch->workDone.store(true);
        batch->workReady.store(false);
    }
}

// LockFreeStateHarvester implementation
inline LockFreeStateHarvester::LockFreeStateHarvester(int numThreads, int obsDim, int envsPerThread)
    : mNumThreads(numThreads), mObsDim(obsDim), mEnvsPerThread(envsPerThread) {
    
    mThreadBuffers.resize(numThreads);
    for (int i = 0; i < numThreads; i++) {
        mThreadBuffers[i].observations.resize(envsPerThread * obsDim);
        mThreadBuffers[i].rewards.resize(envsPerThread);
        mThreadBuffers[i].dones.resize(envsPerThread);
    }
}

inline void LockFreeStateHarvester::StoreFromThread(int threadId, 
                                                     const float* observations, 
                                                     const float* rewards, 
                                                     const bool* dones, 
                                                     int numEnvs) {
    ThreadLocalBuffer& buf = mThreadBuffers[threadId];
    std::memcpy(buf.observations.data(), observations, numEnvs * mObsDim * sizeof(float));
    std::memcpy(buf.rewards.data(), rewards, numEnvs * sizeof(float));
    std::memcpy(buf.dones.data(), dones, numEnvs * sizeof(bool));
    buf.hasData.store(true);
}

inline void LockFreeStateHarvester::MergeAll(float* outObservations, float* outRewards, 
                                              bool* outDones, int obsDim) {
    int offset = 0;
    for (int i = 0; i < mNumThreads; i++) {
        if (mThreadBuffers[i].hasData.load()) {
            int numEnvs = mEnvsPerThread;
            std::memcpy(outObservations + offset * obsDim, 
                        mThreadBuffers[i].observations.data(), 
                        numEnvs * obsDim * sizeof(float));
            std::memcpy(outRewards + offset, 
                        mThreadBuffers[i].rewards.data(), 
                        numEnvs * sizeof(float));
            std::memcpy(outDones + offset, 
                        mThreadBuffers[i].dones.data(), 
                        numEnvs * sizeof(bool));
            offset += numEnvs;
            mThreadBuffers[i].hasData.store(false);
        }
    }
}

inline void LockFreeStateHarvester::Reset() {
    for (int i = 0; i < mNumThreads; i++) {
        mThreadBuffers[i].hasData.store(false);
    }
}

// AsyncActionQueuer implementation
inline AsyncActionQueuer::AsyncActionQueuer(int totalEnvs, int actionDim)
    : mTotalEnvs(totalEnvs), mActionDim(actionDim), mSwapComplete(false) {
    
    mBuffers.buffer0.resize(totalEnvs * actionDim);
    mBuffers.buffer1.resize(totalEnvs * actionDim);
}

inline void AsyncActionQueuer::WriteNextActions(const float* actions, int numActions) {
    int writeIdx = mBuffers.writeBuffer.load();
    float* writeBuffer = (writeIdx == 0) ? mBuffers.buffer0.data() : mBuffers.buffer1.data();
    std::memcpy(writeBuffer, actions, numActions * mActionDim * sizeof(float));
}

inline const float* AsyncActionQueuer::GetCurrentActions(int& numActions) const {
    numActions = mTotalEnvs;
    int readIdx = mBuffers.readBuffer.load();
    return (readIdx == 0) ? mBuffers.buffer0.data() : mBuffers.buffer1.data();
}

inline void AsyncActionQueuer::SignalSwapReady() {
    mBuffers.swapReady.store(true);
    while (!mSwapComplete.load()) {
        _mm_pause();
    }
    mSwapComplete.store(false);
}

inline void AsyncActionQueuer::WaitForSwap() {
    while (!mBuffers.swapReady.load()) {
        _mm_pause();
    }
    
    // Swap buffers
    int newRead = 1 - mBuffers.readBuffer.load();
    mBuffers.readBuffer.store(newRead);
    mBuffers.writeBuffer.store(newRead);
    
    mBuffers.swapReady.store(false);
    mSwapComplete.store(true);
}
