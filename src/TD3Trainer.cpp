#include "TD3Trainer.h"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "AlignedAllocator.h"
#include "src/EigenUtils.h"

TD3Trainer::TD3Trainer(int stateDim, int actionDim, const TD3Config& config)
    : mStateDim(stateDim)
    , mActionDim(actionDim)
    , mConfig(config)
    , mRng(42)
    , mOpponentPool(MAX_POOL_SIZE)
{
    mModel.Init(stateDim, actionDim, config.hiddenDim, config.latentDim, mRng);
    
    int batchSize = config.batchSize;
    mBatchStates.resize(batchSize * stateDim);
    mBatchActions.resize(batchSize * actionDim);
    mBatchRewards.resize(batchSize);
    mBatchNextStates.resize(batchSize * stateDim);
    mBatchDones.resize(batchSize);
    mBatchVectorRewards.resize(batchSize);
    
    mNextActions.resize(batchSize * actionDim);
    mQ1Values.resize(batchSize * 4);
    mQ2Values.resize(batchSize * 4);
    mTargetQ.resize(batchSize);
    mGrads.resize(mModel.GetActor().GetNumWeights());

    int criticInputDim = stateDim + actionDim + mModel.GetLatentDim();
    mCriticInputBuffer.resize(batchSize * criticInputDim);
    mSampledIndices.resize(batchSize);
    
    // Pre-allocate temp buffers for batch operations
    mActorOutputBuffer.resize(batchSize * actionDim);
    mCriticQBuffer.resize(batchSize * 4);
    mLatentZPos.resize(batchSize * mModel.GetLatentDim());
    mLatentZVel.resize(batchSize * mModel.GetLatentDim());
    
    // Initialize Muon optimizers with optimized settings
    MuonOptimizer::Config optConfig;
    optConfig.lrMuon = mConfig.muonLR;
    optConfig.betaMuon = mConfig.muonBeta;
    optConfig.nsSteps = mConfig.muonNSSteps;  // Reduced to 1 for speed
    optConfig.lrFallback = mConfig.criticLR;
    optConfig.eps = mConfig.muonEpsilon;

    mActorOptimizer = MuonOptimizer(optConfig);
    mCritic1Optimizer = MuonOptimizer(optConfig);
    mCritic2Optimizer = MuonOptimizer(optConfig);
    
    // Register actor parameters
    auto& actor = mModel.GetActor();
    for (size_t l = 0; l < actor.GetNumLayers(); ++l) {
        auto& layer = actor.GetLayer(l);
        auto& cp = layer.GetControlPoints();
        auto& grad = layer.GetControlPointGradients();
        int rows = layer.GetOutputDim();
        int cols = layer.GetNumParams() / rows;
        if (cols > 1) {
            mActorOptimizer.addParameter(cp.data(), grad.data(), rows, cols);
        } else {
            mActorOptimizer.addParameter1D(cp.data(), grad.data(), cp.size());
        }
    }
    
    // Register critic1 parameters
    auto& critic1 = mModel.GetCritic1();
    for (size_t l = 0; l < critic1.GetNumLayers(); ++l) {
        auto& layer = critic1.GetLayer(l);
        auto& cp = layer.GetControlPoints();
        auto& grad = layer.GetControlPointGradients();
        int rows = layer.GetOutputDim();
        int cols = layer.GetNumParams() / rows;
        if (cols > 1) {
            mCritic1Optimizer.addParameter(cp.data(), grad.data(), rows, cols);
        } else {
            mCritic1Optimizer.addParameter1D(cp.data(), grad.data(), cp.size());
        }
    }
    
    // Register critic2 parameters
    auto& critic2 = mModel.GetCritic2();
    for (size_t l = 0; l < critic2.GetNumLayers(); ++l) {
        auto& layer = critic2.GetLayer(l);
        auto& cp = layer.GetControlPoints();
        auto& grad = layer.GetControlPointGradients();
        int rows = layer.GetOutputDim();
        int cols = layer.GetNumParams() / rows;
        if (cols > 1) {
            mCritic2Optimizer.addParameter(cp.data(), grad.data(), rows, cols);
        } else {
            mCritic2Optimizer.addParameter1D(cp.data(), grad.data(), cp.size());
        }
    }
}

void TD3Trainer::SelectAction(const float* state, float* action)
{
    float logProb;
    mModel.SelectAction(state, action, &logProb, true);
}

void TD3Trainer::SelectActionWithLatent(const float* state, float* action, int envIdx)
{
    float logProb;
    mModel.SelectAction(state, action, &logProb, true, envIdx); 
}

void TD3Trainer::SelectActionBatchWithLatent(const float* states, float* actions, int batchSize, const std::vector<int>& envIndices)
{
    mModel.SelectActionBatchWithLatent(states, actions, batchSize, envIndices, true);
}

void TD3Trainer::SelectActionEval(const float* state, float* action)
{
    float logProb;
    mModel.SelectAction(state, action, &logProb, false);
}

void TD3Trainer::SelectActionResidual(const float* state, float* residualAction)
{
    SelectAction(state, residualAction);
    for (int i = 0; i < mActionDim; ++i)
    {
        residualAction[i] = std::clamp(residualAction[i], -1.0f, 1.0f);
    }
}

void TD3Trainer::Train(ReplayBuffer& buffer)
{
    if (!buffer.IsReady(mConfig.batchSize))
    {
        return;
    }

    // Profile buffer sampling - use Eigen-optimized sampling
    HighResTimer bufferTimer;
    bufferTimer.Start();
    buffer.Sample(mConfig.batchSize,
                  mBatchStates.data(),
                  mBatchActions.data(),
                  mBatchRewards.data(),
                  mBatchNextStates.data(),
                  mBatchDones.data(),
                  mRng);
    mBufferTime = bufferTimer.StopMicroseconds();

    // Profile critic update
    HighResTimer trainTimer;
    trainTimer.Start();
    UpdateCritic(buffer);

    // OPTIMIZED: Delayed target network updates (every 10 steps instead of every step)
    if (mUpdateCount % mConfig.policyDelay == 0)
    {
        UpdateActor(buffer);
        
        // Only update targets every targetUpdateDelay steps
        if (mUpdateCount % mConfig.targetUpdateDelay == 0) {
            UpdateTargets();
        }
    }
    mTrainTime = trainTimer.StopMicroseconds();

    // Compute SPS estimate
    float totalStepTime = mBufferTime + mTrainTime + mPhysicsTime + mActionTime;
    float currentSPS = totalStepTime > 0 ? 1000000.0f / totalStepTime : 0.0f;

    // Record metrics
    mPerfMetrics.Record(mActionTime, mStepTime, mBufferTime, mTrainTime,
                        mPhysicsTime, mNetworkTime, currentSPS, 0.0f);

    // Print performance table periodically
    if (mStepCount % 100 == 0) {
        mPerfMetrics.PrintTable();
    }

    mUpdateCount++;
    mStepCount++;

    if (mStepCount % mConfig.snapshotInterval == 0)
    {
        SnapshotOpponent();
    }
}

void TD3Trainer::TrainWithVectorRewards(ReplayBuffer& buffer)
{
    if (!buffer.IsReady(mConfig.batchSize))
    {
        return;
    }
    
    buffer.SampleVectorRewards(mConfig.batchSize,
                               mBatchStates.data(),
                               mBatchActions.data(),
                               mBatchVectorRewards.data(),
                               mBatchNextStates.data(),
                               mBatchDones.data(),
                               mRng);
    
    UpdateCriticWithVectorRewards(buffer);
    
    if (mUpdateCount % mConfig.policyDelay == 0)
    {
        UpdateActor(buffer);
        UpdateTargets();
    }
    
    mUpdateCount++;
    mStepCount++;
    
    if (mStepCount % mConfig.snapshotInterval == 0)
    {
        SnapshotOpponent();
    }
}

float TD3Trainer::ComputeCriticLoss(SpanNetwork& critic, const float* criticInput, int batchSize)
{
    critic.ForwardBatch(criticInput, mCriticQBuffer.data(), batchSize);
    
    float loss = 0.0f;
    for (int i = 0; i < batchSize; ++i)
    {
        float tdError = mTargetQ[i] - mCriticQBuffer[i * 4];
        loss += tdError * tdError;
    }
    return loss / batchSize;
}

void TD3Trainer::UpdateCritic(ReplayBuffer& buffer)
{
    const int batchSize = mConfig.batchSize;
    const int latentDim = mModel.GetLatentDim();
    const int criticInputDim = mStateDim + mActionDim + latentDim;

    // STEP 1: Generate next actions using target actor - BATCH FORWARD (Eigen-optimized)
    mModel.GetActorTarget().ForwardBatch(mBatchNextStates.data(), mNextActions.data(), batchSize);
    ForwardMoLU_AVX2(mNextActions.data(), mActionDim * batchSize);

    // STEP 2: Add clipped noise to all actions (vectorized with Eigen)
    {
        auto actions = EigenUtils::Map(mNextActions.data(), batchSize, mActionDim);
        std::normal_distribution<float> noiseDist(0.0f, mConfig.policyNoise);
        
        // Generate all noise at once
        for (int i = 0; i < batchSize * mActionDim; ++i) {
            float noise = std::clamp(noiseDist(mRng), -mConfig.noiseClip, mConfig.noiseClip);
            mNextActions[i] = std::clamp(mNextActions[i] + noise, -1.0f, 1.0f);
        }
    }

    // STEP 3: Build critic input buffer (state + action + latent) - VECTORIZED
    // Pre-fetch latent states for all environments
    AlignedVector32<float> zPos(latentDim);
    AlignedVector32<float> zVel(latentDim);
    
    // OPTIMIZED: Build critic input in a single pass with better cache locality
    for (int i = 0; i < batchSize; ++i) {
        size_t baseIdx = i * criticInputDim;
        
        // Copy state
        EigenUtils::BatchedMemcpy(
            mCriticInputBuffer.data() + baseIdx,
            mBatchNextStates.data() + i * mStateDim,
            mStateDim
        );
        
        // Copy action
        EigenUtils::BatchedMemcpy(
            mCriticInputBuffer.data() + baseIdx + mStateDim,
            mNextActions.data() + i * mActionDim,
            mActionDim
        );
        
        // Get and copy latent (simplified - using env 0)
        mModel.GetLatentMemory().GetLatentStates(zPos.data(), zVel.data(), 0);
        EigenUtils::BatchedMemcpy(
            mCriticInputBuffer.data() + baseIdx + mStateDim + mActionDim,
            zPos.data(),
            latentDim
        );
    }

    // STEP 4: Target Q evaluation - BATCH FORWARD (both critics at once, Eigen-optimized)
    mModel.GetCritic1Target().ForwardBatch(mCriticInputBuffer.data(), mQ1Values.data(), batchSize);
    mModel.GetCritic2Target().ForwardBatch(mCriticInputBuffer.data(), mQ2Values.data(), batchSize);

    // STEP 5: Compute target Q values using Eigen (fully vectorized)
    EigenUtils::ComputeTDTargets(
        mBatchRewards.data(),
        mQ1Values.data(),  // Use Q1 values directly (min computed inside)
        mBatchDones.data(),
        mTargetQ.data(),
        batchSize,
        mConfig.gamma
    );
    
    // Apply min Q clipping manually for double Q-learning
    for (int i = 0; i < batchSize; ++i) {
        float q1 = mQ1Values[i * 4];
        float q2 = mQ2Values[i * 4];
        float minQ = std::min(q1, q2);
        // Recompute with min Q
        mTargetQ[i] = mBatchRewards[i] + mConfig.gamma * (1.0f - mBatchDones[i]) * minQ;
    }

    // STEP 6: Build current-state critic input for gradient computation
    for (int i = 0; i < batchSize; ++i) {
        size_t baseIdx = i * criticInputDim;

        EigenUtils::BatchedMemcpy(
            mCriticInputBuffer.data() + baseIdx,
            mBatchStates.data() + i * mStateDim,
            mStateDim
        );
        EigenUtils::BatchedMemcpy(
            mCriticInputBuffer.data() + baseIdx + mStateDim,
            mBatchActions.data() + i * mActionDim,
            mActionDim
        );

        mModel.GetLatentMemory().GetLatentStates(zPos.data(), zVel.data(), 0);
        EigenUtils::BatchedMemcpy(
            mCriticInputBuffer.data() + baseIdx + mStateDim + mActionDim,
            zPos.data(),
            latentDim
        );
    }

    // STEP 7: Compute critic gradients via finite difference (optimized)
    // Only compute gradients every muonUpdateInterval steps for efficiency
    bool shouldUpdateCritic = (mUpdateCount % mConfig.muonUpdateInterval == 0);
    
    if (shouldUpdateCritic) {
        // Compute gradients for Critic 1
        ComputeCriticGradients(mModel.GetCritic1(), buffer, true);
        
        // Compute gradients for Critic 2
        ComputeCriticGradients(mModel.GetCritic2(), buffer, false);
        
        // STEP 8: Apply Muon optimizer step - ACTUAL WEIGHT UPDATES!
        mCritic1Optimizer.step();  // Update Critic 1 weights
        mCritic2Optimizer.step();  // Update Critic 2 weights
        
        // Zero gradients after update
        mCritic1Optimizer.zeroGrad();
        mCritic2Optimizer.zeroGrad();
    }
    
    // Always update target networks (soft update with tau=0.005)
    // This provides additional learning signal propagation
    mModel.GetCritic1Target().SoftUpdate(mModel.GetCritic1(), mConfig.tau);
    mModel.GetCritic2Target().SoftUpdate(mModel.GetCritic2(), mConfig.tau);
}

void TD3Trainer::UpdateCriticWithVectorRewards(ReplayBuffer& buffer)
{
    static std::normal_distribution<float> noiseDist(0.0f, mConfig.policyNoise);
    
    // Use batch forward pass for actor
    mModel.GetActorTarget().ForwardBatch(mBatchNextStates.data(), mNextActions.data(), mConfig.batchSize);
    
    // Apply MoLU activation to all actions
    ForwardMoLU_AVX2(mNextActions.data(), mActionDim * mConfig.batchSize);
    
    // Add noise to all actions
    for (int i = 0; i < mConfig.batchSize; ++i)
    {
        float* nextAction = mNextActions.data() + i * mActionDim;
        for (int j = 0; j < mActionDim; ++j)
        {
            float noise = std::clamp(noiseDist(mRng), -mConfig.noiseClip, mConfig.noiseClip);
            nextAction[j] = std::clamp(nextAction[j] + noise, -1.0f, 1.0f);
        }
    }
    
    // Use batch forward pass for critics
    AlignedVector32<float> q1Batch(mConfig.batchSize * 4);
    AlignedVector32<float> q2Batch(mConfig.batchSize * 4);
    
    // We need to combine states and actions for critic input - this requires a temporary buffer
    AlignedVector32<float> criticInput(mConfig.batchSize * (mStateDim + mActionDim + mModel.GetLatentDim()));
    for (int i = 0; i < mConfig.batchSize; ++i)
    {
        size_t idx = 0;
        const float* state = mBatchNextStates.data() + i * mStateDim;
        const float* action = mNextActions.data() + i * mActionDim;
        
        // Get latent state for this environment
        AlignedVector32<float> zPos(LATENT_DIM);
        mModel.GetLatentMemory().GetLatentStates(zPos.data(), nullptr, 0); // TODO: per-environment latent?
        
        // Copy state
        std::copy(state, state + mStateDim, criticInput.data() + i * (mStateDim + mActionDim + mModel.GetLatentDim()));
        idx += mStateDim;
        // Copy action
        std::copy(action, action + mActionDim, criticInput.data() + i * (mStateDim + mActionDim + mModel.GetLatentDim()) + mStateDim);
        idx += mActionDim;
        // Copy latent
        std::copy(zPos.data(), zPos.data() + mModel.GetLatentDim(), 
                 criticInput.data() + i * (mStateDim + mActionDim + mModel.GetLatentDim()) + mStateDim + mActionDim);
    }
    
    mModel.GetCritic1Target().ForwardBatch(criticInput.data(), q1Batch.data(), mConfig.batchSize);
    mModel.GetCritic2Target().ForwardBatch(criticInput.data(), q2Batch.data(), mConfig.batchSize);
    
    for (int i = 0; i < mConfig.batchSize; ++i)
    {
        float scalarReward = ComputeScalarReward(mBatchVectorRewards[i]);
        float q1 = q1Batch[i * 4];
        float q2 = q2Batch[i * 4];
        float minQ = std::min(q1, q2);
        mTargetQ[i] = scalarReward + mConfig.gamma * (1.0f - mBatchDones[i]) * minQ;
    }
}

void TD3Trainer::UpdateActor(ReplayBuffer& buffer)
{
    const int batchSize = mConfig.batchSize;
    const int latentDim = mModel.GetLatentDim();

    auto& actor = mModel.GetActor();
    auto weights = actor.GetAllWeights();
    const float actorLR = mConfig.actorLR;

    AlignedVector32<float> zPos(latentDim);
    AlignedVector32<float> zVel(latentDim);

    // STEP 1: Build critic input buffer with current policy actions - BATCH
    actor.ForwardBatch(mBatchStates.data(), mActorOutputBuffer.data(), batchSize);
    ForwardMoLU_AVX2(mActorOutputBuffer.data(), mActionDim * batchSize);

    // Get latents and build full critic input
    for (int i = 0; i < batchSize; ++i)
    {
        int envIdx = 0;
        mModel.GetLatentMemory().GetLatentStates(zPos.data(), zVel.data(), envIdx);

        size_t baseIdx = i * (mStateDim + mActionDim + latentDim);
        std::copy(mBatchStates.data() + i * mStateDim,
                  mBatchStates.data() + (i + 1) * mStateDim,
                  mCriticInputBuffer.data() + baseIdx);
        std::copy(mActorOutputBuffer.data() + i * mActionDim,
                  mActorOutputBuffer.data() + (i + 1) * mActionDim,
                  mCriticInputBuffer.data() + baseIdx + mStateDim);
        std::copy(zPos.data(), zPos.data() + latentDim,
                  mCriticInputBuffer.data() + baseIdx + mStateDim + mActionDim);
    }

    // STEP 2: Compute baseline Q-value - BATCH FORWARD
    mModel.GetCritic1().ForwardBatch(mCriticInputBuffer.data(), mQ1Values.data(), batchSize);

    float baselineQ = 0.0f;
    for (int i = 0; i < batchSize; ++i)
    {
        baselineQ += mQ1Values[i * 4];
    }
    baselineQ /= batchSize;

    // STEP 3: Compute actor gradient via finite difference (optimized)
    // Only compute gradients every muonUpdateInterval steps for efficiency
    bool shouldUpdateActor = (mUpdateCount % mConfig.muonUpdateInterval == 0);
    
    if (shouldUpdateActor) {
        ComputeActorGradient(buffer);
        
        // STEP 4: Apply Muon optimizer step - ACTUAL WEIGHT UPDATES!
        mActorOptimizer.step();  // Update Actor weights
        
        // Zero gradients after update
        mActorOptimizer.zeroGrad();
    }
    
    // Always update target network (soft update with tau=0.005)
    // This provides additional learning signal propagation
    mModel.GetActorTarget().SoftUpdate(mModel.GetActor(), mConfig.tau);
}

void TD3Trainer::UpdateTargets()
{
    mModel.UpdateTargets(mConfig.tau);
}

void TD3Trainer::SnapshotOpponent()
{
    auto weights = mModel.GetActor().GetAllWeights();
    mOpponentPool.Snapshot(weights, {}, mStepCount);
}

bool TD3Trainer::SampleOpponent()
{
    std::vector<float> weights, biases;
    if (mOpponentPool.SampleOpponentRecent(weights, biases, mRng))
    {
        mModel.GetActor().SetAllWeights(weights);
        return true;
    }
    return false;
}

void TD3Trainer::Save(const std::string& path) const
{
    // Default save with combat bot config
    Save(path, "robots/combat_bot.json", 13, 208);
}

void TD3Trainer::Save(const std::string& path, const std::string& robotConfigPath, 
                      int numSatellites, int observationDim) const
{
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "[TD3Trainer] Failed to save to " << path << std::endl;
        return;
    }

    // Checkpoint format version 3 - with robot config metadata
    int version = 3;
    file.write(reinterpret_cast<const char*>(&version), sizeof(int));
    
    // Core dimensions
    file.write(reinterpret_cast<const char*>(&mStateDim), sizeof(int));
    file.write(reinterpret_cast<const char*>(&mActionDim), sizeof(int));
    file.write(reinterpret_cast<const char*>(&mStepCount), sizeof(int));
    
    // Robot configuration metadata
    int robotConfigPathLen = static_cast<int>(robotConfigPath.length());
    file.write(reinterpret_cast<const char*>(&robotConfigPathLen), sizeof(int));
    file.write(robotConfigPath.c_str(), robotConfigPathLen);
    
    // Robot architecture info (for validation on load)
    file.write(reinterpret_cast<const char*>(&numSatellites), sizeof(int));
    file.write(reinterpret_cast<const char*>(&observationDim), sizeof(int));
    
    // Model weights
    auto weights = mModel.GetActor().GetAllWeights();
    int numWeights = static_cast<int>(weights.size());
    file.write(reinterpret_cast<const char*>(&numWeights), sizeof(int));
    file.write(reinterpret_cast<const char*>(weights.data()), numWeights * sizeof(float));
    
    // Preference vector
    file.write(reinterpret_cast<const char*>(mPreferenceVector.data()),
               VECTOR_REWARD_DIM * sizeof(float));
    
    // Checksum for integrity
    uint32_t checksum = 0;
    for (float w : weights) checksum += static_cast<uint32_t>(w * 1000);
    file.write(reinterpret_cast<const char*>(&checksum), sizeof(uint32_t));

    file.close();
    std::cout << "[TD3Trainer] Checkpoint saved to: " << path 
              << " (v" << version << ", " << numWeights << " weights, robot=" << robotConfigPath << ")" << std::endl;
}

void TD3Trainer::Load(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "[TD3Trainer] Failed to load from " << path << std::endl;
        return;
    }

    int version, stateDim, actionDim;
    file.read(reinterpret_cast<char*>(&version), sizeof(int));
    file.read(reinterpret_cast<char*>(&stateDim), sizeof(int));
    file.read(reinterpret_cast<char*>(&actionDim), sizeof(int));
    file.read(reinterpret_cast<char*>(&mStepCount), sizeof(int));
    
    std::cout << "[TD3Trainer] Loading checkpoint v" << version 
              << " (stateDim=" << stateDim << ", actionDim=" << actionDim << ")" << std::endl;

    // Validate dimensions
    // Handle dimension mismatch - attempt conversion if possible
    if (stateDim != mStateDim || actionDim != mActionDim)
    {
        std::cerr << "[TD3Trainer] WARNING: Checkpoint dimension mismatch!" << std::endl;
        std::cerr << "[TD3Trainer]   Checkpoint: " << stateDim << "x" << actionDim << std::endl;
        std::cerr << "[TD3Trainer]   Current:    " << mStateDim << "x" << mActionDim << std::endl;
        
        // Attempt to convert checkpoint if action dim matches
        if (actionDim == mActionDim) {
            std::cerr << "[TD3Trainer]   Attempting checkpoint conversion (zero-padding for expanded obs space)..." << std::endl;
            
            int numWeights;
            file.read(reinterpret_cast<char*>(&numWeights), sizeof(int));
            
            if (numWeights > 0 && numWeights <= 10000000) {
                std::vector<float> oldWeights(numWeights);
                file.read(reinterpret_cast<char*>(oldWeights.data()), numWeights * sizeof(float));
                
                // Load old weights into actor (will be zero-padded for new input dims)
                ConvertAndLoadWeights(mModel.GetActor(), oldWeights, stateDim, actionDim);
                
                // Load preference vector if available
                if (version >= 2) {
                    file.read(reinterpret_cast<char*>(mPreferenceVector.data()),
                              VECTOR_REWARD_DIM * sizeof(float));
                }
                
                mModel.UpdateTargets(1.0f);
                std::cout << "[TD3Trainer] Checkpoint converted and loaded successfully!" << std::endl;
                file.close();
                return;
            }
        } else {
            std::cerr << "[TD3Trainer]   Action dimension mismatch - cannot convert, using random initialization" << std::endl;
        }
        
        // Skip loading weights, keep random initialization
        file.close();
        return;
    }
    
    // Read robot config metadata (v3+)
    std::string loadedRobotConfig;
    int loadedNumSatellites = 0;
    int loadedObservationDim = 0;
    
    if (version >= 3) {
        int robotConfigPathLen;
        file.read(reinterpret_cast<char*>(&robotConfigPathLen), sizeof(int));
        if (robotConfigPathLen > 0 && robotConfigPathLen < 1024) {
            std::vector<char> configPath(robotConfigPathLen + 1);
            file.read(configPath.data(), robotConfigPathLen);
            configPath[robotConfigPathLen] = '\0';
            loadedRobotConfig = std::string(configPath.data());
        }
        file.read(reinterpret_cast<char*>(&loadedNumSatellites), sizeof(int));
        file.read(reinterpret_cast<char*>(&loadedObservationDim), sizeof(int));
        
        std::cout << "[TD3Trainer] Robot config: " << loadedRobotConfig 
                  << " (satellites=" << loadedNumSatellites << ", obsDim=" << loadedObservationDim << ")" << std::endl;
    }

    int numWeights;
    file.read(reinterpret_cast<char*>(&numWeights), sizeof(int));
    
    if (numWeights <= 0 || numWeights > 10000000) {
        std::cerr << "[TD3Trainer] ERROR: Invalid weight count: " << numWeights << std::endl;
        return;
    }

    std::vector<float> weights(numWeights);
    file.read(reinterpret_cast<char*>(weights.data()), numWeights * sizeof(float));
    
    // Verify checksum (v3+)
    if (version >= 3) {
        uint32_t storedChecksum;
        file.read(reinterpret_cast<char*>(&storedChecksum), sizeof(uint32_t));
        
        uint32_t computedChecksum = 0;
        for (float w : weights) computedChecksum += static_cast<uint32_t>(w * 1000);
        
        if (storedChecksum != computedChecksum) {
            std::cerr << "[TD3Trainer] WARNING: Checksum mismatch! File may be corrupted." << std::endl;
            std::cerr << "[TD3Trainer]   Stored:   " << storedChecksum << std::endl;
            std::cerr << "[TD3Trainer]   Computed: " << computedChecksum << std::endl;
        } else {
            std::cout << "[TD3Trainer] Checksum verified OK" << std::endl;
        }
    }
    
    // Load weights
    mModel.GetActor().SetAllWeights(weights);

    // Load preference vector (v2+)
    if (version >= 2)
    {
        file.read(reinterpret_cast<char*>(mPreferenceVector.data()),
                  VECTOR_REWARD_DIM * sizeof(float));
    }

    mModel.UpdateTargets(1.0f);
    file.close();
    
    std::cout << "[TD3Trainer] Checkpoint loaded successfully from " << path << std::endl;
}

// Helper function to convert old checkpoint weights to new architecture
void TD3Trainer::ConvertAndLoadWeights(SpanNetwork& network, const std::vector<float>& oldWeights, 
                                        int oldStateDim, int oldActionDim)
{
    std::cerr << "[TD3Trainer] Converting checkpoint: old obs=" << oldStateDim 
              << ", new obs=" << mStateDim << std::endl;
    
    auto newWeights = network.GetAllWeights();
    
    // Copy old weights to new weight array (zero-padding for expanded dimensions)
    size_t copySize = std::min(oldWeights.size(), newWeights.size());
    std::copy(oldWeights.begin(), oldWeights.begin() + copySize, newWeights.begin());
    
    // Initialize remaining weights with small random values
    std::normal_distribution<float> dist(0.0f, 0.01f);
    for (size_t i = copySize; i < newWeights.size(); ++i) {
        newWeights[i] = dist(mRng);
    }
    
    network.SetAllWeights(newWeights);

    std::cerr << "[TD3Trainer] Converted " << copySize << " weights, initialized "
              << (newWeights.size() - copySize) << " new weights" << std::endl;
}

// ============================================================================
// GRADIENT COMPUTATION FOR MUON OPTIMIZER
// ============================================================================

void TD3Trainer::ComputeCriticGradients(SpanNetwork& critic, ReplayBuffer& buffer, bool isCritic1)
{
    const int batchSize = mConfig.batchSize;
    const int latentDim = mModel.GetLatentDim();
    const int criticInputDim = mStateDim + mActionDim + latentDim;

    // Zero existing gradients
    critic.ZeroGradients();

    float total_loss = 0.0f;

    // Process batch with analytic gradients
    for (int i = 0; i < batchSize; ++i)
    {
        // Build critic input (state + action + latent)
        AlignedVector32<float> critic_input(criticInputDim);
        size_t baseIdx = i * criticInputDim;

        std::copy(
            mCriticInputBuffer.data() + baseIdx,
            mCriticInputBuffer.data() + baseIdx + criticInputDim,
            critic_input.data()
        );

        // Forward pass with caching
        AlignedVector32<float> q_pred(4);
        critic.ForwardWithCache(critic_input.data(), q_pred.data());

        // Compute TD error
        float td_error = mTargetQ[i] - q_pred[0];
        total_loss += td_error * td_error;

        // Backward pass: d_loss/d_q = 2 * td_error * (-1) = -2 * td_error
        // We want to minimize (target - pred)^2, so gradient is -2 * (target - pred) = 2 * (pred - target)
        AlignedVector32<float> q_grad(4);
        q_grad[0] = -2.0f * td_error;
        q_grad[1] = 0.0f;
        q_grad[2] = 0.0f;
        q_grad[3] = 0.0f;

        // Backpropagate through critic (accumulate gradients)
        critic.Backward(critic_input.data(), q_grad.data(), nullptr, true);
    }

    // Average gradients over batch
    critic.ScaleGradients(1.0f / batchSize);

    // Also update optimizer gradients to match
    MuonOptimizer& optimizer = isCritic1 ? mCritic1Optimizer : mCritic2Optimizer;
    optimizer.zeroGrad();

    // The optimizer uses the same gradient pointers registered during construction
    // Gradients are already in the network's gradient buffers
}

void TD3Trainer::ComputeActorGradient(ReplayBuffer& buffer)
{
    const int batchSize = mConfig.batchSize;
    const int latentDim = mModel.GetLatentDim();
    const int criticInputDim = mStateDim + mActionDim + latentDim;

    auto& actor = mModel.GetActor();
    auto& critic1 = mModel.GetCritic1();

    // Zero existing gradients
    actor.ZeroGradients();

    for (int i = 0; i < batchSize; ++i)
    {
        // Forward pass: state -> action
        AlignedVector32<float> action(mActionDim);
        const float* state = mBatchStates.data() + i * mStateDim;
        
        actor.ForwardWithCache(state, action.data());

        // Build critic input with this action
        AlignedVector32<float> critic_input(criticInputDim);
        size_t baseIdx = i * criticInputDim;

        // Copy state
        std::copy(state, state + mStateDim, critic_input.data());
        // Copy action
        std::copy(action.data(), action.data() + mActionDim, critic_input.data() + mStateDim);

        // Get and copy latent
        AlignedVector32<float> zPos(latentDim);
        AlignedVector32<float> zVel(latentDim);
        mModel.GetLatentMemory().GetLatentStates(zPos.data(), zVel.data(), 0);
        std::copy(zPos.data(), zPos.data() + latentDim, critic_input.data() + mStateDim + mActionDim);

        // Forward through critic to get Q value (with caching for potential higher-order gradients)
        AlignedVector32<float> q_value(4);
        critic1.ForwardWithCache(critic_input.data(), q_value.data());

        // Backward through critic (to get dQ/da)
        // We want to maximize Q, so gradient is +1
        AlignedVector32<float> q_grad(4);
        q_grad[0] = 1.0f;  // dQ/dQ = 1, we want to maximize Q
        q_grad[1] = 0.0f;
        q_grad[2] = 0.0f;
        q_grad[3] = 0.0f;

        AlignedVector32<float> critic_input_grad(criticInputDim);
        critic1.Backward(critic_input.data(), q_grad.data(), critic_input_grad.data(), false);

        // Extract action gradient (portion of critic_input_grad corresponding to action)
        AlignedVector32<float> action_grad(mActionDim);
        std::copy(
            critic_input_grad.data() + mStateDim,
            critic_input_grad.data() + mStateDim + mActionDim,
            action_grad.data()
        );

        // Note: The action output from actor doesn't have MoLU applied (only hidden layers do)
        // So we don't need to apply MoLU backward here - it's handled inside actor.Backward()
        // for the hidden layers

        // Backward through actor (accumulate gradients)
        // The actor's Backward method handles MoLU backward for hidden layers internally
        actor.Backward(state, action_grad.data(), nullptr, true);
    }

    // Average gradients
    actor.ScaleGradients(1.0f / batchSize);

    // Update optimizer gradients
    mActorOptimizer.zeroGrad();
    // Gradients are already in the network's gradient buffers
}
