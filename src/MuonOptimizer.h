#pragma once

#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

class MuonOptimizer {
public:
    struct Config {
        float lrMuon;
        float betaMuon;
        int nsSteps;
        float lrFallback;
        float betaFallback;
        float eps;

        Config() : lrMuon(0.02f), betaMuon(0.95f), nsSteps(1),  // Reduced nsSteps from 3 to 1 for speed
                   lrFallback(0.001f), betaFallback(0.9f), eps(1e-8f) {}
    };

    MuonOptimizer(const Config& config = Config());

    // Newton-Schulz polynomial constants for fast orthogonalization
    static constexpr float NS_A = 3.4445f;
    static constexpr float NS_B = -4.7750f;
    static constexpr float NS_C = 2.3315f;

    void addParameter(float* paramPtr, float* gradPtr, int rows, int cols);
    void addParameter1D(float* paramPtr, float* gradPtr, int size);
    void step();
    void zeroGrad();
    
    // Set configuration dynamically
    void setNSSteps(int steps) { mConfig.nsSteps = steps; }
    void setLearningRate(float lr) { mConfig.lrMuon = lr; }

private:
    Config mConfig;

    struct MatrixParam {
        float* ptr;
        float* gradPtr;
        int rows, cols;
        Eigen::MatrixXf momentum;
    };

    struct VectorParam {
        float* ptr;
        float* gradPtr;
        int size;
        Eigen::VectorXf momentum;
    };

    std::vector<MatrixParam> mMatrixParams;
    std::vector<VectorParam> mVectorParams;

    // Optimized orthogonalization with single Newton-Schulz iteration
    Eigen::MatrixXf orthogonalize(const Eigen::MatrixXf& G);
    
    // Parallel matrix orthogonalization for batch processing
    void orthogonalizeBatch(std::vector<Eigen::MatrixXf>& matrices, 
                           std::vector<Eigen::MatrixXf>& results);
};

constexpr float NS_A = 3.4445f;
constexpr float NS_B = -4.7750f;
constexpr float NS_C = 2.0315f;
