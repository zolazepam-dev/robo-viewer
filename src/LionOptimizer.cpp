#include "LionOptimizer.h"
#include <algorithm>
#include <cmath>

LionOptimizer::LionOptimizer(const Config& config) : mConfig(config) {}

void LionOptimizer::addParameter(float* paramPtr, float* gradPtr, int size) {
    Parameter param;
    param.ptr = paramPtr;
    param.gradPtr = gradPtr;
    param.size = size;
    param.momentum = Eigen::VectorXf::Zero(size);
    mParams.push_back(param);
}

void LionOptimizer::addParameter1D(float* paramPtr, float* gradPtr, int size) {
    // Same as addParameter - kept for API compatibility
    addParameter(paramPtr, gradPtr, size);
}

void LionOptimizer::step() {
    // Lion update rule:
    // m_t = beta * m_{t-1} + (1 - beta) * g_t
    // w_t = w_{t-1} - lr * sign(m_t)
    
    for (auto& param : mParams) {
        Eigen::Map<Eigen::VectorXf> gradMap(param.gradPtr, param.size);
        
        // Update momentum: m = beta * m + (1 - beta) * g
        param.momentum = mConfig.beta * param.momentum + (1.0f - mConfig.beta) * gradMap;
        
        // Compute sign of momentum and apply update
        // sign(x) = 1 if x > 0, -1 if x < 0, 0 if x == 0
        Eigen::VectorXf update = param.momentum.array().sign();
        
        // Apply weight decay if configured
        if (mConfig.weightDecay > 0.0f) {
            Eigen::Map<Eigen::VectorXf> paramMap(param.ptr, param.size);
            update += mConfig.weightDecay * paramMap;
        }
        
        // Apply update: w = w - lr * sign(m)
        Eigen::Map<Eigen::VectorXf> paramMap(param.ptr, param.size);
        paramMap -= mConfig.lr * update;
    }
}

void LionOptimizer::zeroGrad() {
    for (auto& param : mParams) {
        std::fill(param.gradPtr, param.gradPtr + param.size, 0.0f);
    }
}
