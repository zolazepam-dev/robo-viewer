// STRICT REQUIREMENT: Jolt.h must be included first
#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>

#include <chrono>
#include <iostream>
#include <random>
#include <mutex>

#include "VectorizedEnv.h"
#include "TD3Trainer.h"

// Minimal end-to-end smoke test: initialize physics, roll environments,
// insert transitions, and perform at least one trainer update. Keeps
// iterations tiny to remain fast while verifying the pipeline wiring.
int main()
{
    // Jolt factory registration (thread-safe once)
    static std::once_flag joltInitFlag;
    std::call_once(joltInitFlag, []() {
        JPH::RegisterDefaultAllocator();
        JPH::Factory::sInstance = new JPH::Factory();
        JPH::RegisterTypes();
    });

    const int kEnvs = 4;
    VectorizedEnv vecEnv(kEnvs);
    vecEnv.Init();

    const int stateDim = vecEnv.GetObservationDim();
    const int actionDim = vecEnv.GetActionDim();
    const int totalActionDim = actionDim * 2 * kEnvs;

    TD3Config cfg;
    cfg.hiddenDim = 64;
    cfg.batchSize = 4;
    cfg.startSteps = 2; // train almost immediately
    cfg.bufferSize = 256;

    TD3Trainer trainer(stateDim, actionDim, cfg);
    ReplayBuffer buffer(cfg.bufferSize, stateDim, actionDim);

    AlignedVector32<float> actions(totalActionDim, 0.0f);
    std::mt19937 rng(123);
    std::normal_distribution<float> noise(0.0f, 0.2f);

    const int kSteps = 8;
    for (int step = 0; step < kSteps; ++step)
    {
        // Random exploratory actions to keep the test deterministic but non-zero
        for (int i = 0; i < totalActionDim; ++i) actions[i] = noise(rng);

        vecEnv.Step(actions);
        vecEnv.ResetDoneEnvs();

        const auto& obs = vecEnv.GetObservations();
        const auto& rewards = vecEnv.GetVectorRewards();
        const auto& dones = vecEnv.GetDones();

        for (int envIdx = 0; envIdx < kEnvs; ++envIdx)
        {
            const float* s1 = obs.data() + envIdx * stateDim * 2;
            const float* s2 = s1 + stateDim;
            const float* a1 = actions.data() + envIdx * actionDim * 2;
            const float* a2 = a1 + actionDim;

            buffer.Add(s1, a1, rewards[envIdx], s2, dones[envIdx]);
            buffer.Add(s2, a2, rewards[envIdx], s1, dones[envIdx]);
        }

        if (buffer.Size() >= cfg.startSteps)
        {
            trainer.TrainWithVectorRewards(buffer);
        }
    }

    if (buffer.Size() == 0)
    {
        std::cerr << "[E2E] Buffer is empty after rollout" << std::endl;
        return 1;
    }

    // Quick dimensional sanity checks
    if (stateDim <= 0 || actionDim <= 0)
    {
        std::cerr << "[E2E] Invalid dimensions stateDim=" << stateDim
                  << " actionDim=" << actionDim << std::endl;
        return 1;
    }

    std::cout << "[E2E] Success: rollout=" << kSteps
              << " buffer_size=" << buffer.Size()
              << " stateDim=" << stateDim
              << " actionDim=" << actionDim << std::endl;

    return 0;
}
