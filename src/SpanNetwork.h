#pragma once

#include <vector>
#include <random>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstring>

#include "NeuralMath.h"
#include "LatentMemory.h"
#include "RFFNetwork.h"
#include "RFFLatentDynamics.h"
#include "AlignedAllocator.h"

// Re-export RFF types for backward compatibility
using SpanLayerConfig = RFFLayerConfig;
using RFFConfig = RFFConfig;

// Alias TensorProductBSpline to RFFLayer for compatibility (deprecated)
using TensorProductBSpline = RFFLayer;

// Alias SpanNetwork to RFFNetwork for compatibility
using SpanNetwork = RFFNetwork;

// Re-export SpanCache from RFFNetwork
using SpanCache = ::SpanCache;

// Alias SpanActorCritic to RFFActorCritic for compatibility
using SpanActorCritic = RFFActorCritic;

// Re-export CriticBatchBuffer from original (kept for compatibility)
struct alignas(32) CriticBatchBuffer
{
    AlignedVector32<float> preActivation;
    AlignedVector32<float> postActivation;
    AlignedVector32<float> gradients;

    AlignedVector32<float> weights;
    AlignedVector32<float> biases;

    void Init(size_t hiddenDim, size_t batchSize)
    {
        size_t hiddenAligned = GetAVX2PaddedSize(hiddenDim);
        preActivation.assign(batchSize * hiddenAligned, 0.0f);
        postActivation.assign(batchSize * hiddenAligned, 0.0f);
        gradients.assign(batchSize * hiddenAligned, 0.0f);
        weights.assign(hiddenAligned * hiddenAligned, 0.0f);
        biases.assign(hiddenAligned, 0.0f);
    }

    void Clear()
    {
        std::fill(preActivation.begin(), preActivation.end(), 0.0f);
        std::fill(postActivation.begin(), postActivation.end(), 0.0f);
        std::fill(gradients.begin(), gradients.end(), 0.0f);
    }
};
