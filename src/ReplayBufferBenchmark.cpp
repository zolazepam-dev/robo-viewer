#include "modules/replay/ReplayBuffer.h"
#include "modules/common/PerformanceDiagnoser.h"
#include <iostream>
#include <vector>
#include <random>

int main() {
    const int capacity = 100000; // Small enough to fit in memory
    const int obsDim = 256;
    const int actionDim = 32;
    const int batchSize = 256;
    const int numSteps = 5000;

    std::cout << "[Benchmark] Initializing ReplayBuffer SoA (Capacity: " << capacity << ")..." << std::endl;
    ReplayBuffer buffer(capacity, obsDim, actionDim);
    
    std::mt19937 rng(42);
    std::vector<float> obs(obsDim, 1.0f);
    std::vector<float> action(actionDim, 0.5f);
    
    std::cout << "[Benchmark] Starting Add/Sample Loop (" << numSteps << " iterations)..." << std::endl;
    
    float* outStates = new float[batchSize * obsDim * 2];
    float* outActions = new float[batchSize * actionDim * 2];
    float* outRewards = new float[batchSize];
    float* outNextStates = new float[batchSize * obsDim * 2];
    float* outDones = new float[batchSize];

    for (int i = 0; i < numSteps; ++i) {
        // Simulating 128 environments adding data
        for (int e = 0; e < 128; ++e) {
            buffer.Add(obs.data(), obs.data(), action.data(), action.data(), 1.0f, 1.0f, obs.data(), obs.data(), false);
        }
        
        if (buffer.Size() >= batchSize) {
            buffer.Sample(batchSize, outStates, outActions, outRewards, outNextStates, outDones, rng);
        }
        
        if (i % 1000 == 0) {
            std::cout << "Iteration " << i << " completed..." << std::endl;
        }
    }

    PerformanceDiagnoser::Get().PrintReport();
    
    delete[] outStates;
    delete[] outActions;
    delete[] outRewards;
    delete[] outNextStates;
    delete[] outDones;

    return 0;
}
