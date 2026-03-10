// SpanNetwork.cpp - Compatibility wrapper for RFF-based architecture
// This file exists for backward compatibility. The actual implementation
// is in RFFNetwork.cpp and RFFLayer.cpp

#include "SpanNetwork.h"

// All functionality is now provided by RFFNetwork and RFFLayer
// This file is kept for build system compatibility

// Note: The following type aliases are defined in SpanNetwork.h:
// - SpanNetwork -> RFFNetwork
// - SpanActorCritic -> RFFActorCritic
// - TensorProductBSpline -> RFFLayer
// - SpanLayerConfig -> RFFLayerConfig

// The RFF-based implementation provides:
// 1. Faster forward passes (no B-spline basis computation)
// 2. Simpler gradient computation (only linear layer backprop)
// 3. Same interface for TD3Trainer compatibility
// 4. Preserved latent ODE dynamics for temporal memory
