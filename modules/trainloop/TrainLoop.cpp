/**
 * @file TrainLoop.cpp
 * @brief Implementation of TrainLoop class
 */

#include "TrainLoop.h"
#include "../physics/PhysicsWorld.h"
#include "../vectorenv/VectorEnv.h"
#include "../replay/ReplayBuffer.h"
#include "../visualizer/Visualizer.h"
#include "../agent/TD3Trainer.h"
#include <iostream>
#include <thread>
#include <filesystem>

namespace fs = std::filesystem;

TrainLoop::TrainLoop(const TrainConfig& config)
    : mConfig(config)
    , mRunning(false)
    , mStopRequested(false)
{
}

TrainLoop::~TrainLoop() {
    Stop();
}

bool TrainLoop::Init() {
    std::cout << "[TrainLoop] Initializing..." << std::endl;
    
    // Create checkpoint directory
    fs::create_directories(mConfig.checkpointDir);
    
    // Initialize physics world
    mPhysicsWorld = std::make_unique<PhysicsWorld>();
    
    // Thread pinning configuration (optional)
    const int pinnedCores[] = {1, 2, 3, 4, 5, 7, 8, 9, 10, 11};
    if (!mPhysicsWorld->Init(mConfig.numEnvs, 10, pinnedCores)) {
        std::cerr << "[TrainLoop] Failed to initialize physics world" << std::endl;
        return false;
    }
    
    // Initialize vectorized environments
    mVectorEnv = std::make_unique<VectorEnv>(mConfig.numEnvs, mConfig.stepsPerEpisode);
    mVectorEnv->Init(mPhysicsWorld.get(), true);
    
    // Initialize replay buffer
    int obsDim = mVectorEnv->GetObservationDim();
    int actDim = mVectorEnv->GetActionDim();
    mReplayBuffer = std::make_unique<ReplayBuffer>(
        mConfig.bufferCapacity, obsDim, actDim
    );
    
    // Initialize trainer
    mTrainer = std::make_unique<TD3Trainer>(obsDim, actDim);
    
    // Try to load existing model
    std::string modelPath = mConfig.checkpointDir + "/model_final.bin";
    if (!mConfig.loadModelPath.empty()) {
        modelPath = mConfig.loadModelPath;
    }
    
    if (fs::exists(modelPath)) {
        mTrainer->Load(modelPath);
        std::cout << "[TrainLoop] Loaded model from: " << modelPath << std::endl;
    }
    
    // Initialize visualizer (if not headless)
    if (!mConfig.headlessTurbo) {
        mVisualizer = std::make_unique<Visualizer>(1280, 720);
        if (!mVisualizer->Init()) {
            std::cerr << "[TrainLoop] Failed to initialize visualizer" << std::endl;
            return false;
        }
    }
    
    mLastTime = std::chrono::high_resolution_clock::now();
    mLastCheckpoint = mLastTime;
    mLastStatsUpdate = mLastTime;
    
    std::cout << "[TrainLoop] Initialization complete" << std::endl;
    return true;
}

void TrainLoop::Run() {
    if (!mPhysicsWorld || !mVectorEnv || !mTrainer || !mReplayBuffer) {
        std::cerr << "[TrainLoop] Run() called before Init()" << std::endl;
        return;
    }
    
    mRunning = true;
    mStopRequested = false;
    
    std::cout << "[TrainLoop] Starting training loop" << std::endl;
    
    while (mRunning && !mStopRequested) {
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - mLastTime).count();
        mLastTime = now;
        
        // Process user commands (from visualizer UI)
        if (mVisualizer) {
            mVisualizer->PollEvents();
            if (mVisualizer->ShouldClose()) {
                mStopRequested = true;
            }
            ProcessCommands();
        }
        
        // Step environments
        StepEnvironments();
        
        // Train agent
        if (mConfig.trainEnabled && mStats.totalSteps > mConfig.startSteps) {
            TrainAgent();
        }
        
        // Update statistics
        UpdateStats();
        
        // Checkpoint periodically
        Checkpoint();
        
        // Reset done environments
        ResetDoneEnvs();
        
        // Render (if visualizer enabled)
        if (mVisualizer && !mConfig.headlessTurbo) {
            UserCommands commands;
            mVisualizer->Render(mPhysicsWorld.get(), commands);
            mVisualizer->SwapBuffers();
        }
        
        mStepCounter++;
        mStats.totalSteps++;
    }
    
    mRunning = false;
    std::cout << "[TrainLoop] Training loop ended" << std::endl;
}

void TrainLoop::Stop() {
    mStopRequested = true;
    while (mRunning) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void TrainLoop::ProcessCommands() {
    if (!mVisualizer) return;
    
    const UserCommands& cmds = mVisualizer->GetCommands();
    
    if (cmds.reset) {
        mVectorEnv->Reset();
        mVisualizer->ClearReset();
    }
    
    if (cmds.stepOne) {
        StepEnvironments();
        mVisualizer->ClearStepOne();
    }
    
    std::string modelName;
    if (mVisualizer->GetAndClearSaveRequest(modelName)) {
        std::string path = mConfig.checkpointDir + "/" + modelName + ".bin";
        mTrainer->Save(path);
        std::cout << "[TrainLoop] Saved model to: " << path << std::endl;
    }
    
    if (mVisualizer->GetAndClearLoadRequest(modelName)) {
        std::string path = mConfig.checkpointDir + "/" + modelName + ".bin";
        if (fs::exists(path)) {
            mTrainer->Load(path);
            std::cout << "[TrainLoop] Loaded model from: " << path << std::endl;
        }
    }
}

void TrainLoop::StepEnvironments() {
    if (!mVectorEnv || !mTrainer) return;
    
    int totalActions = mVectorEnv->GetTotalActionSize();
    std::vector<float> actions(totalActions);
    
    // Get actions from trainer
    const float* obs = mVectorEnv->GetObservations();
    for (int i = 0; i < mConfig.numEnvs; ++i) {
        float* envActions = actions.data() + (i * 2 * mVectorEnv->GetActionDim());
        float* obs1 = const_cast<float*>(obs) + (i * 2 * mVectorEnv->GetObservationDim());
        float* obs2 = obs1 + mVectorEnv->GetObservationDim();
        
        mTrainer->SelectAction(obs1, envActions);
        mTrainer->SelectAction(obs2, envActions + mVectorEnv->GetActionDim());
    }
    
    // Step all environments
    mVectorEnv->Step(actions.data());
}

void TrainLoop::TrainAgent() {
    if (!mTrainer || !mReplayBuffer) return;
    
    // Sample from replay buffer and train
    if (mReplayBuffer->CanSample(mConfig.batchSize)) {
        mTrainer->Train(*mReplayBuffer);
    }
}

void TrainLoop::Checkpoint() {
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - mLastCheckpoint).count();
    
    if (elapsed >= mConfig.checkpointInterval / 100) {  // Check every ~100 steps
        std::string path = mConfig.checkpointDir + "/model_final.bin";
        mTrainer->Save(path);
        mLastCheckpoint = now;
    }
}

void TrainLoop::UpdateStats() {
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<float>(now - mLastStatsUpdate).count();
    
    if (elapsed >= 1.0f) {  // Update every second
        mStats.sps = static_cast<float>(mStepCounter - mStepCounterReset) / elapsed;
        mStepCounterReset = mStepCounter;
        
        // Get rewards from vector env
        if (mVectorEnv) {
            const float* rewards = mVectorEnv->GetRewards();
            const char* dones = mVectorEnv->GetDones();
            
            float totalReward = 0.0f;
            int doneCount = 0;
            
            for (int i = 0; i < mConfig.numEnvs; ++i) {
                if (dones[i] != 0) {
                    totalReward += rewards[i * 2] + rewards[i * 2 + 1];
                    doneCount++;
                }
            }
            
            if (doneCount > 0) {
                mStats.avgReward = totalReward / doneCount;
                mStats.episodes += doneCount;
            }
        }
        
        mLastStatsUpdate = now;
    }
}

void TrainLoop::ResetDoneEnvs() {
    if (mVectorEnv) {
        mVectorEnv->ResetDoneEnvs();
    }
}
