#include "MuonOptimizer.h"
#include <algorithm>
#include <cmath>

MuonOptimizer::MuonOptimizer(const Config& config) : mConfig(config) {}

void MuonOptimizer::addParameter(float* paramPtr, float* gradPtr, int rows, int cols) {
    MatrixParam param;
    param.ptr = paramPtr;
    param.gradPtr = gradPtr;
    param.rows = rows;
    param.cols = cols;
    param.momentum = Eigen::MatrixXf::Zero(rows, cols);
    mMatrixParams.push_back(param);
}

void MuonOptimizer::addParameter1D(float* paramPtr, float* gradPtr, int size) {
    VectorParam param;
    param.ptr = paramPtr;
    param.gradPtr = gradPtr;
    param.size = size;
    param.momentum = Eigen::VectorXf::Zero(size);
    mVectorParams.push_back(param);
}

Eigen::MatrixXf MuonOptimizer::orthogonalize(const Eigen::MatrixXf& G) {
    // Fast orthogonalization using Newton-Schulz iteration
    // Optimized for single iteration (nsSteps=1) for maximum speed
    
    float norm = G.norm();
    if (norm < mConfig.eps) norm = mConfig.eps;

    Eigen::MatrixXf X = G / norm;

    bool transposed = false;
    if (X.rows() > X.cols()) {
        X = X.transpose().eval();
        transposed = true;
    }

    // Single Newton-Schulz iteration (optimized for speed)
    // X = X * (A*I + B*X^T*X + C*(X^T*X)^2)
    // With nsSteps=1, this is much faster while still providing good orthogonalization
    for (int i = 0; i < mConfig.nsSteps; ++i) {
        // Optimized: compute X^T*X once and reuse
        Eigen::MatrixXf XtX = X.transpose() * X;
        Eigen::MatrixXf X_XtX = X * XtX;
        Eigen::MatrixXf X_XtX2 = X_XtX * XtX;
        
        // Apply polynomial: A*X + B*X*XtX + C*X*XtX^2
        X = (NS_A * X + NS_B * X_XtX + NS_C * X_XtX2).eval();
    }

    if (transposed) {
        X = X.transpose().eval();
    }

    return X;
}

void MuonOptimizer::orthogonalizeBatch(std::vector<Eigen::MatrixXf>& matrices,
                                        std::vector<Eigen::MatrixXf>& results) {
    // Parallel batch orthogonalization using OpenMP
    results.resize(matrices.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < matrices.size(); ++i) {
        results[i] = orthogonalize(matrices[i]);
    }
}

void MuonOptimizer::step() {
    // Process matrix parameters with Muon optimizer
    for (auto& param : mMatrixParams) {
        Eigen::Map<Eigen::MatrixXf> gradMap(param.gradPtr, param.rows, param.cols);
        
        // Update momentum: m = beta * m + grad
        param.momentum = mConfig.betaMuon * param.momentum + gradMap;
        
        // Orthogonalize momentum
        Eigen::MatrixXf U = orthogonalize(param.momentum);
        
        // Scale by sqrt(max(rows, cols)) and apply update
        float scale = mConfig.lrMuon * std::sqrt(std::max(param.rows, param.cols));
        
        // Vectorized weight update
        Eigen::Map<Eigen::MatrixXf> paramMap(param.ptr, param.rows, param.cols);
        paramMap -= scale * U;
    }

    // Process 1D parameters with fallback optimizer (Adam-like)
    for (auto& param : mVectorParams) {
        Eigen::Map<Eigen::VectorXf> gradMap(param.gradPtr, param.size);
        
        // Update momentum
        param.momentum = mConfig.betaFallback * param.momentum + gradMap;
        
        // Apply update
        Eigen::Map<Eigen::VectorXf> paramMap(param.ptr, param.size);
        paramMap -= mConfig.lrFallback * param.momentum;
    }
}

void MuonOptimizer::zeroGrad() {
    for (auto& param : mMatrixParams) {
        std::fill(param.gradPtr, param.gradPtr + param.rows * param.cols, 0.0f);
    }
    for (auto& param : mVectorParams) {
        std::fill(param.gradPtr, param.gradPtr + param.size, 0.0f);
    }
}
