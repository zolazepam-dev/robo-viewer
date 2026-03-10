#pragma once

#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

/**
 * Lion Optimizer (Evolved Sign Momentum)
 * 
 * Much faster than Muon - no expensive Newton-Schulz orthogonalization!
 * 
 * Update rule:
 *   m_t = beta * m_{t-1} + (1 - beta) * g_t
 *   w_t = w_{t-1} - lr * sign(m_t)
 * 
 * References:
 *   "Symbolic Discovery of Optimization Algorithms" (2023)
 *   https://arxiv.org/abs/2302.06675
 */
class LionOptimizer {
public:
    struct Config {
        float lr;           // Learning rate (default: 1e-4)
        float beta;         // Momentum decay (default: 0.9)
        float weightDecay;  // Weight decay (default: 0.0)
        
        Config() : lr(1e-4f), beta(0.9f), weightDecay(0.0f) {}
    };
    
    LionOptimizer(const Config& config = Config());
    
    void addParameter(float* paramPtr, float* gradPtr, int size);
    void addParameter1D(float* paramPtr, float* gradPtr, int size);  // For biases
    void step();
    void zeroGrad();
    
    // Set learning rate dynamically
    void setLearningRate(float lr) { mConfig.lr = lr; }
    float getLearningRate() const { return mConfig.lr; }
    
private:
    Config mConfig;
    
    struct Parameter {
        float* ptr;
        float* gradPtr;
        int size;
        Eigen::VectorXf momentum;
    };
    
    std::vector<Parameter> mParams;
};
