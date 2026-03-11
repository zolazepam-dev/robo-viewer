#pragma once

#include "SpanNetwork.h"
#include "AlignedAllocator.h"

#include <random>
#include <cmath>
#include <algorithm>
#include <vector>

class DopamineOptimizer {
public:
    struct Config {
        float baseLearningRate = 3e-4f;
        float rpeThreshold = 0.01f;
        float maxLearningRate = 1e-2f;
        float minLearningRate = 1e-6f;
        int perturbationStride = 64;
        bool useStochasticUpdate = true;
        float noiseScale = 0.05f;
    };

    DopamineOptimizer(const Config& config = Config(), std::mt19937* rng = nullptr)
        : mConfig(config), mRng(rng ? *rng : std::mt19937(42)) {}

    template<typename NetworkType>
    int Optimize(NetworkType& network, 
                 const float* inputBatch, 
                 int batchSize,
                 float* targetValues,
                 float* currentQValues,
                 float* gradients = nullptr,
                 int numGradients = 0) {
        
        auto weights = network.GetAllWeights();
        const int numWeights = static_cast<int>(weights.size());
        
        if (numWeights == 0) return 0;
        
        float baselineLoss = ComputeLoss(network, inputBatch, batchSize, targetValues);
        float baselineQ = ComputeAverageQ(currentQValues, batchSize);
        
        int updates = 0;
        
        // Create indices for perturbation
        std::vector<int> weightIndices;
        weightIndices.reserve(numWeights / mConfig.perturbationStride);
        
        if (mConfig.useStochasticUpdate) {
            std::uniform_int_distribution<int> dist(0, numWeights - 1);
            for (int i = 0; i < numWeights / mConfig.perturbationStride; ++i) {
                int idx = dist(mRng);
                weightIndices.push_back(idx);
            }
        } else {
            for (int w = 0; w < numWeights; w += mConfig.perturbationStride) {
                weightIndices.push_back(w);
            }
        }
        
        for (size_t i = 0; i < weightIndices.size(); ++i) {
            int idx = weightIndices[i];
            if (idx >= numWeights) continue;
            
            float rpe = ComputeRewardPredictionError(network, weights, idx, inputBatch, batchSize, targetValues, baselineQ);
            float adaptiveLR = CalculateAdaptiveLR(rpe);
            
            float originalWeight = weights[idx];
            float noise = std::normal_distribution<float>(0.0f, mConfig.noiseScale)(mRng);
            float candidateWeight = originalWeight + adaptiveLR * noise;
            
            weights[idx] = candidateWeight;
            network.SetAllWeights(weights);
            
            float candidateLoss = ComputeLoss(network, inputBatch, batchSize, targetValues);
            float candidateQ = ComputeAverageQ(currentQValues, batchSize);
            
            if (candidateLoss <= baselineLoss || candidateQ >= baselineQ) {
                updates++;
                baselineLoss = candidateLoss;
                baselineQ = candidateQ;
            } else {
                weights[idx] = originalWeight;
                network.SetAllWeights(weights);
            }
        }
        
        return updates;
    }

private:
    Config mConfig;
    std::mt19937 mRng;
    
    template<typename NetworkType>
    float ComputeLoss(NetworkType& network, const float* inputBatch, int batchSize, const float* targetValues) {
        AlignedVector32<float> outputs(batchSize * 4);
        network.ForwardBatch(inputBatch, outputs.data(), batchSize);
        
        float loss = 0.0f;
        for (int i = 0; i < batchSize; ++i) {
            float tdError = targetValues[i] - outputs[i * 4];
            loss += tdError * tdError;
        }
        return loss / batchSize;
    }
    
    float ComputeAverageQ(const float* qValues, int batchSize) {
        if (batchSize <= 0) return 0.0f;
        float sum = 0.0f;
        for (int i = 0; i < batchSize; ++i) {
            sum += qValues[i];
        }
        return sum / batchSize;
    }
    
    template<typename NetworkType>
    float ComputeRewardPredictionError(NetworkType& network,
                                      const std::vector<float>& weights,
                                      int weightIdx,
                                      const float* inputBatch,
                                      int batchSize,
                                      const float* targetValues,
                                      float baselineQ) {
        float currentWeight = weights[weightIdx];
        float epsilon = 1e-3f;
        float perturbedWeight = currentWeight + epsilon;
        
        float baselineLoss = ComputeLoss(network, inputBatch, batchSize, targetValues);
        
        std::vector<float> tempWeights = weights;
        tempWeights[weightIdx] = perturbedWeight;
        network.SetAllWeights(tempWeights);
        
        float perturbedLoss = ComputeLoss(network, inputBatch, batchSize, targetValues);
        network.SetAllWeights(weights);
        
        float expectedImprovement = -(perturbedLoss - baselineLoss);
        float actualImprovement = 0.0f;
        
        AlignedVector32<float> outputs(batchSize * 4);
        network.ForwardBatch(inputBatch, outputs.data(), batchSize);
        float perturbedQ = ComputeAverageQ(outputs.data(), batchSize);
        
        actualImprovement = perturbedQ - baselineQ;
        
        return expectedImprovement - actualImprovement;
    }
    
    float CalculateAdaptiveLR(float rpe) {
        float absRPE = std::abs(rpe);
        float scaledRPE = 1.0f / (1.0f + std::exp(-absRPE * 10.0f));
        float lr = mConfig.baseLearningRate * (1.0f + scaledRPE * 5.0f);
        return std::max(mConfig.minLearningRate, std::min(lr, mConfig.maxLearningRate));
    }
};