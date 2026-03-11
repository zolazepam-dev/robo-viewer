#include <gtest/gtest.h>
#include "RFFTensor.hpp"
#include <vector>
#include <iostream>

using namespace cput;
using namespace cput::rff;

class RFFTensorTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(RFFTensorTest, Initialization) {
    RFFTensorLayer layer;
    layer.init(10, 128, 1.0f);
    
    EXPECT_EQ(layer.getWeights().shape(0), 1);
    EXPECT_EQ(layer.getWeights().shape(1), 128);
    EXPECT_EQ(layer.getBias().shape(0), 1);
}

TEST_F(RFFTensorTest, ForwardPass) {
    RFFTensorLayer layer;
    size_t in_dim = 10;
    size_t rff_dim = 64;
    layer.init(in_dim, rff_dim, 1.0f);
    
    size_t batch_size = 4;
    TensorF32 input(batch_size, in_dim);
    input.fill(1.0f);
    
    TensorF32 output;
    layer.forward(input, output);
    
    EXPECT_EQ(output.shape(0), batch_size);
    EXPECT_EQ(output.shape(1), 1);
    
    // Check that output is not zero
    float sum = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        sum += std::abs(output.data()[i]);
    }
    EXPECT_GT(sum, 0.0f);
    
    std::cout << "RFF Tensor Output (first 2): " << output.data()[0] << ", " << output.data()[1] << std::endl;
}

TEST_F(RFFTensorTest, BatchConsistency) {
    RFFTensorLayer layer;
    size_t in_dim = 5;
    size_t rff_dim = 32;
    layer.init(in_dim, rff_dim, 1.0f);
    
    // Single sample
    TensorF32 input1(1, in_dim);
    input1.fill(0.5f);
    TensorF32 output1;
    layer.forward(input1, output1);
    
    // Batch of 2 same samples
    TensorF32 input2(2, in_dim);
    input2.fill(0.5f);
    TensorF32 output2;
    layer.forward(input2, output2);
    
    EXPECT_NEAR(output1.data()[0], output2.data()[0], 1e-5f);
    EXPECT_NEAR(output1.data()[0], output2.data()[1], 1e-5f);
}
