/**
 * @file neural_net.hpp
 * @brief Neural Network Components for Multi-Agent RL
 * 
 * This module provides:
 * - Matrix/vector operations (Eigen-based or manual)
 * - Fully connected layer with forward/backward passes
 * - Actor network (policy - outputs action probabilities)
 * - Critic network (value function)
 * - Activation functions (ReLU, Tanh, Softmax)
 */

#ifndef MARL_NEURAL_NET_HPP
#define MARL_NEURAL_NET_HPP

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <iomanip>
#include <sstream>
#include <memory>

#ifdef USE_EIGEN
#include <Eigen/Dense>
#endif

namespace marl {

// ============================================================================
// Math Utilities
// ============================================================================

/**
 * @brief Simple vector/matrix operations without external dependencies
 */
class Math {
public:
    using Vector = std::vector<double>;
    using Matrix = std::vector<std::vector<double>>;
    
    /**
     * @brief Create zero-initialized vector
     */
    static Vector zeros(size_t size) {
        return Vector(size, 0.0);
    }
    
    /**
     * @brief Create zero-initialized matrix
     */
    static Matrix zeros(size_t rows, size_t cols) {
        return Matrix(rows, Vector(cols, 0.0));
    }
    
    /**
     * @brief Create random matrix with Xavier initialization
     */
    static Matrix random(size_t rows, size_t cols, double std_dev = 0.01) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0.0, std_dev);
        
        Matrix m(rows, Vector(cols));
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                m[i][j] = d(gen);
            }
        }
        return m;
    }
    
    /**
     * @brief Matrix-vector multiplication
     */
    static Vector matmul(const Matrix& m, const Vector& v) {
        assert(m[0].size() == v.size());
        Vector result(m.size(), 0.0);
        
        for (size_t i = 0; i < m.size(); i++) {
            double sum = 0.0;
            for (size_t j = 0; j < m[i].size(); j++) {
                sum += m[i][j] * v[j];
            }
            result[i] = sum;
        }
        return result;
    }
    
    /**
     * @brief Element-wise addition
     */
    static Vector add(const Vector& a, const Vector& b) {
        assert(a.size() == b.size());
        Vector result(a.size());
        for (size_t i = 0; i < a.size(); i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }
    
    /**
     * @brief Element-wise multiplication (Hadamard product)
     */
    static Vector multiply(const Vector& a, const Vector& b) {
        assert(a.size() == b.size());
        Vector result(a.size());
        for (size_t i = 0; i < a.size(); i++) {
            result[i] = a[i] * b[i];
        }
        return result;
    }
    
    /**
     * @brief Scalar multiplication
     */
    static Vector scale(const Vector& v, double scalar) {
        Vector result(v.size());
        for (size_t i = 0; i < v.size(); i++) {
            result[i] = v[i] * scalar;
        }
        return result;
    }
    
    /**
     * @brief ReLU activation
     */
    static Vector relu(const Vector& v) {
        Vector result(v.size());
        for (size_t i = 0; i < v.size(); i++) {
            result[i] = std::max(0.0, v[i]);
        }
        return result;
    }
    
    /**
     * @brief ReLU derivative
     */
    static Vector relu_derivative(const Vector& v) {
        Vector result(v.size());
        for (size_t i = 0; i < v.size(); i++) {
            result[i] = v[i] > 0 ? 1.0 : 0.0;
        }
        return result;
    }
    
    /**
     * @brief Tanh activation
     */
    static Vector tanh(const Vector& v) {
        Vector result(v.size());
        for (size_t i = 0; i < v.size(); i++) {
            result[i] = std::tanh(v[i]);
        }
        return result;
    }
    
    /**
     * @brief Tanh derivative
     */
    static Vector tanh_derivative(const Vector& v) {
        Vector result(v.size());
        for (size_t i = 0; i < v.size(); i++) {
            double t = std::tanh(v[i]);
            result[i] = 1.0 - t * t;
        }
        return result;
    }
    
    /**
     * @brief Softmax activation (for action probabilities)
     */
    static Vector softmax(const Vector& v) {
        Vector result(v.size());
        double max_val = *std::max_element(v.begin(), v.end());
        double sum = 0.0;
        
        for (size_t i = 0; i < v.size(); i++) {
            result[i] = std::exp(v[i] - max_val);
            sum += result[i];
        }
        
        for (size_t i = 0; i < v.size(); i++) {
            result[i] /= sum;
        }
        
        return result;
    }
    
    /**
     * @brief Sum of vector elements
     */
    static double sum(const Vector& v) {
        return std::accumulate(v.begin(), v.end(), 0.0);
    }
    
    /**
     * @brief Dot product
     */
    static double dot(const Vector& a, const Vector& b) {
        assert(a.size() == b.size());
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
    
    /**
     * @brief Matrix transpose-vector multiplication: m^T * v
     */
    static Vector matmul_transpose(const Matrix& m, const Vector& v) {
        // m is stored as [rows][cols], we want m^T * v
        // Result has size cols
        if (m.empty()) return Vector();
        
        Vector result(m[0].size(), 0.0);
        for (size_t j = 0; j < m[0].size(); j++) {
            double sum = 0.0;
            for (size_t i = 0; i < m.size(); i++) {
                sum += m[i][j] * v[i];
            }
            result[j] = sum;
        }
        return result;
    }
    
    /**
     * @brief Outer product (for gradient computation)
     */
    static Matrix outer(const Vector& a, const Vector& b) {
        Matrix result(a.size(), Vector(b.size()));
        for (size_t i = 0; i < a.size(); i++) {
            for (size_t j = 0; j < b.size(); j++) {
                result[i][j] = a[i] * b[j];
            }
        }
        return result;
    }
};

// ============================================================================
// Fully Connected Layer
// ============================================================================

/**
 * @brief Fully connected neural network layer with bias
 */
class FullyConnectedLayer {
public:
    using Vector = Math::Vector;
    using Matrix = Math::Matrix;
    
    /**
     * @brief Initialize layer with given dimensions
     * @param input_dim Input dimension
     * @param output_dim Output dimension
     * @param activation Activation function ("relu", "tanh", "none")
     */
    FullyConnectedLayer(size_t input_dim, size_t output_dim, 
                       const std::string& activation = "relu")
        : m_input_dim(input_dim)
        , m_output_dim(output_dim)
        , m_activation(activation) {
        
        // Xavier initialization
        double std_dev = std::sqrt(2.0 / (input_dim + output_dim));
        m_weights = Math::random(input_dim, output_dim, std_dev);
        m_bias = Math::zeros(output_dim);
        
        // Initialize gradients
        m_weight_grads = Math::zeros(input_dim, output_dim);
        m_bias_grads = Math::zeros(output_dim);
        
        // Cache for backward pass
        m_last_input = Math::zeros(input_dim);
        m_last_output = Math::zeros(output_dim);
        m_last_pre_activation = Math::zeros(output_dim);
    }
    
    /**
     * @brief Forward pass
     * @param input Input vector
     * @return Output vector
     */
    Vector forward(const Vector& input) {
        assert(input.size() == m_input_dim);
        
        // Cache input for backward pass
        m_last_input = input;
        
        // Linear transformation: z = Wx + b
        m_last_pre_activation = Math::add(Math::matmul_transpose(m_weights, input), m_bias);
        
        // Apply activation
        if (m_activation == "relu") {
            m_last_output = Math::relu(m_last_pre_activation);
        } else if (m_activation == "tanh") {
            m_last_output = Math::tanh(m_last_pre_activation);
        } else {
            m_last_output = m_last_pre_activation;
        }
        
        return m_last_output;
    }
    
    /**
     * @brief Backward pass - compute gradients
     * @param grad_output Gradient from next layer
     * @return Gradient w.r.t. input
     */
    Vector backward(const Vector& grad_output) {
        assert(grad_output.size() == m_output_dim);
        
        // Gradient through activation
        Vector grad_pre_activation;
        if (m_activation == "relu") {
            grad_pre_activation = Math::multiply(
                grad_output, 
                Math::relu_derivative(m_last_pre_activation)
            );
        } else if (m_activation == "tanh") {
            grad_pre_activation = Math::multiply(
                grad_output,
                Math::tanh_derivative(m_last_pre_activation)
            );
        } else {
            grad_pre_activation = grad_output;
        }
        
        // Gradient w.r.t. weights: dW = x * grad^T
        m_weight_grads = Math::outer(m_last_input, grad_pre_activation);
        
        // Gradient w.r.t. bias: db = grad
        m_bias_grads = grad_pre_activation;
        
        // Gradient w.r.t. input: dx = W * grad
        Vector grad_input = Math::matmul(m_weights, grad_pre_activation);
        
        return grad_input;
    }
    
    /**
     * @brief Update weights using gradient descent
     * @param lr Learning rate
     */
    void update(double lr) {
        // Update weights
        for (size_t i = 0; i < m_weights.size(); i++) {
            for (size_t j = 0; j < m_weights[i].size(); j++) {
                m_weights[i][j] -= lr * m_weight_grads[i][j];
            }
        }
        
        // Update biases
        for (size_t i = 0; i < m_bias.size(); i++) {
            m_bias[i] -= lr * m_bias_grads[i];
        }
    }
    
    /**
     * @brief Get current weights
     */
    const Matrix& get_weights() const { return m_weights; }
    
    /**
     * @brief Get current biases
     */
    const Vector& get_bias() const { return m_bias; }
    
    /**
     * @brief Set weights (for loading/saving)
     */
    void set_weights(const Matrix& w) { m_weights = w; }
    
    /**
     * @brief Set biases
     */
    void set_bias(const Vector& b) { m_bias = b; }
    
    /**
     * @brief Get input dimension
     */
    size_t input_dim() const { return m_input_dim; }
    
    /**
     * @brief Get output dimension
     */
    size_t output_dim() const { return m_output_dim; }

private:
    size_t m_input_dim;
    size_t m_output_dim;
    std::string m_activation;
    
    Matrix m_weights;
    Vector m_bias;
    Matrix m_weight_grads;
    Vector m_bias_grads;
    
    Vector m_last_input;
    Vector m_last_output;
    Vector m_last_pre_activation;
};

// ============================================================================
// Actor Network (Policy)
// ============================================================================

/**
 * @brief Actor network that outputs action probabilities
 * 
 * Architecture: Input -> Hidden (ReLU) -> Output (Softmax)
 */
class ActorNetwork {
public:
    using Vector = Math::Vector;
    
    /**
     * @brief Initialize actor network
     * @param state_dim Observation/state dimension
     * @param action_dim Number of possible actions
     * @param hidden_dim Hidden layer dimension
     */
    ActorNetwork(size_t state_dim, size_t action_dim, size_t hidden_dim = 64)
        : m_state_dim(state_dim)
        , m_action_dim(action_dim) {
        
        // Two-layer network
        m_layer1 = std::make_unique<FullyConnectedLayer>(state_dim, hidden_dim, "relu");
        m_layer2 = std::make_unique<FullyConnectedLayer>(hidden_dim, action_dim, "none");
    }
    
    /**
     * @brief Forward pass - get action probabilities
     * @param state Input state vector
     * @return Action probability distribution
     */
    Vector forward(const Vector& state) {
        assert(state.size() == m_state_dim);
        
        Vector hidden = m_layer1->forward(state);
        Vector logits = m_layer2->forward(hidden);
        
        // Apply softmax for probabilities
        m_last_probs = Math::softmax(logits);
        return m_last_probs;
    }
    
    /**
     * @brief Sample action from policy
     * @param state Input state
     * @return Sampled action index
     */
    int sample_action(const Vector& state) {
        Vector probs = forward(state);
        
        // Sample from categorical distribution
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> d(probs.begin(), probs.end());
        
        return d(gen);
    }
    
    /**
     * @brief Get action probabilities (without sampling)
     */
    const Vector& get_last_probs() const { return m_last_probs; }
    
    /**
     * @brief Backward pass for policy gradient
     * @param state Input state
     * @param action Taken action index
     * @param advantage Policy gradient advantage
     */
    void backward(const Vector& state, int action, double advantage) {
        // Policy gradient: grad = -advantage * grad_log_pi(a|s)
        // For softmax output: grad_logits = probs - one_hot(action)
        
        Vector grad_logits = m_last_probs;
        grad_logits[action] -= 1.0;  // One-hot subtraction
        grad_logits = Math::scale(grad_logits, advantage);
        
        // Backprop through layers
        Vector grad_hidden = m_layer2->backward(grad_logits);
        m_layer1->backward(grad_hidden);
    }
    
    /**
     * @brief Update network weights
     * @param lr Learning rate
     */
    void update(double lr) {
        m_layer1->update(lr);
        m_layer2->update(lr);
    }
    
    /**
     * @brief Get state dimension
     */
    size_t state_dim() const { return m_state_dim; }
    
    /**
     * @brief Get action dimension
     */
    size_t action_dim() const { return m_action_dim; }

private:
    size_t m_state_dim;
    size_t m_action_dim;
    Vector m_last_probs;
    
    std::unique_ptr<FullyConnectedLayer> m_layer1;
    std::unique_ptr<FullyConnectedLayer> m_layer2;
};

// ============================================================================
// Critic Network (Value Function)
// ============================================================================

/**
 * @brief Critic network that estimates state value
 * 
 * Architecture: Input -> Hidden (ReLU) -> Output (scalar value)
 */
class CriticNetwork {
public:
    using Vector = Math::Vector;
    
    /**
     * @brief Initialize critic network
     * @param state_dim Observation/state dimension
     * @param hidden_dim Hidden layer dimension
     */
    CriticNetwork(size_t state_dim, size_t hidden_dim = 64)
        : m_state_dim(state_dim) {
        
        // Two-layer network, output is scalar
        m_layer1 = std::make_unique<FullyConnectedLayer>(state_dim, hidden_dim, "relu");
        m_layer2 = std::make_unique<FullyConnectedLayer>(hidden_dim, 1, "none");
    }
    
    /**
     * @brief Forward pass - estimate state value
     * @param state Input state vector
     * @return Estimated value V(s)
     */
    double forward(const Vector& state) {
        assert(state.size() == m_state_dim);
        
        Vector hidden = m_layer1->forward(state);
        Vector output = m_layer2->forward(hidden);
        
        m_last_value = output[0];
        return m_last_value;
    }
    
    /**
     * @brief Get last estimated value
     */
    double get_last_value() const { return m_last_value; }
    
    /**
     * @brief Backward pass for value prediction error
     * @param state Input state
     * @param td_error Temporal difference error
     */
    void backward(const Vector& state, double td_error) {
        // Gradient for MSE loss: dL/d_output = 2 * (pred - target) = 2 * td_error
        // We use td_error directly (factor of 2 absorbed into learning rate)
        Vector grad_output = {td_error};
        
        Vector grad_hidden = m_layer2->backward(grad_output);
        m_layer1->backward(grad_hidden);
    }
    
    /**
     * @brief Update network weights
     * @param lr Learning rate
     */
    void update(double lr) {
        m_layer1->update(lr);
        m_layer2->update(lr);
    }
    
    /**
     * @brief Get state dimension
     */
    size_t state_dim() const { return m_state_dim; }

private:
    size_t m_state_dim;
    double m_last_value = 0.0;
    
    std::unique_ptr<FullyConnectedLayer> m_layer1;
    std::unique_ptr<FullyConnectedLayer> m_layer2;
};

} // namespace marl

#endif // MARL_NEURAL_NET_HPP
