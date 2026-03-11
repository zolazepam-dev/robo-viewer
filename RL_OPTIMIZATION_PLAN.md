# JOLTrl RL Training Pipeline Optimization Plan

## Goal: Maximize Steps Per Second (SPS) through structural improvements

**Constraints:**
- 1 rendered environment + N headless environments
- No reduction in brain size or training effectiveness
- Focus on: batching, vectorization, parallelization, memory optimizations

---

## Current Architecture Analysis

### Strengths Already Present:
1. ✅ AVX2-optimized activations (MoLU, Tanh, ReLU)
2. ✅ 32-byte aligned memory (AlignedAllocator)
3. ✅ SpanNetwork with batched forward pass
4. ✅ VectorizedEnv for multiple parallel environments
5. ✅ Lock-free action buffer
6. ✅ SIMD observation packing
7. ✅ OpenMP parallelization in training loop
8. ✅ Muon optimizer with analytic gradients

### Bottlenecks Identified:

1. **Neural Network Forward Pass** - Sequential layer-by-layer execution
2. **Environment Stepping** - Physics systems step sequentially
3. **Memory Layout** - Array of Structures (AoS) in some places
4. **Cache Efficiency** - Missing prefetching opportunities
5. **Thread Utilization** - Limited parallelism in data collection
6. **Batch Size** - Could be larger for better throughput
7. **Latent Memory** - Not fully vectorized across environments

---

## Optimization Phases

### Phase 1: Neural Network Throughput ⚡

#### 1.1 Layer Fusion
**Problem:** Each layer writes to memory then reads back for next layer
**Solution:** Fuse consecutive operations to keep data in registers/cache

```cpp
// Before: Separate forward + activation
layer.ForwardBatch(input, temp, batchSize);
ForwardMoLU_AVX2(temp, size);

// After: Fused operation
layer.ForwardWithActivationBatch(input, output, batchSize, MoLU);
```

**Expected Gain:** 15-25% faster forward pass

#### 1.2 Batched Matrix Multiplication with Eigen/ARMADILLO
**Problem:** Manual AVX2 matmul doesn't utilize CPU cache optimally
**Solution:** Use Eigen with AVX2 vectorization and cache blocking

```cpp
// Use Eigen with aligned allocators and explicit vectorization
Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
    input_map(input, batchSize, inputDim);
Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
    weights_map(weights, inputDim, outputDim);
output_map.noalias() = input_map * weights_map;
```

**Expected Gain:** 20-40% faster GEMM

#### 1.3 Persistent Activation Caching
**Problem:** Recomputing activations during backward pass
**Solution:** Cache all intermediate activations during forward pass

```cpp
struct ForwardCache {
    std::vector<AlignedVector32<float>> layerInputs;   // Input to each layer
    std::vector<AlignedVector32<float>> layerOutputs;  // Output after activation
    std::vector<AlignedVector32<float>> preActivations; // Before activation (for backward)
};
```

**Expected Gain:** 30-50% faster backward pass

---

### Phase 2: Parallel Environment Stepping 🔄

#### 2.1 Multi-Physics System Parallelization
**Problem:** All environments share single physics system (sequential)
**Solution:** Split environments across multiple physics systems

```cpp
class ParallelPhysicsStepper {
    std::vector<PhysicsCore> physicsSystems;  // One per thread
    std::vector<std::thread> workerThreads;
    
    void StepAllParallel() {
        // Each thread steps its physics system independently
        #pragma omp parallel for
        for (int i = 0; i < numPhysicsSystems; i++) {
            physicsSystems[i].Step(timestep);
        }
    }
};
```

**Expected Gain:** Near-linear scaling with CPU cores (8x on 8-core)

#### 2.2 Lock-Free State Harvesting
**Problem:** HarvestStates uses locks for thread safety
**Solution:** Use per-thread buffers, then merge

```cpp
struct ThreadLocalHarvest {
    AlignedVector32<float> observations;  // Per-thread buffer
    AlignedVector32<float> rewards;
    std::vector<bool> dones;
    char padding[64];  // Cache line padding to avoid false sharing
};

std::vector<ThreadLocalHarvest, 64> threadBuffers;  // Aligned to cache line
```

**Expected Gain:** 2-3x faster state harvesting

#### 2.3 Speculative Action Queuing
**Problem:** Actions queued after physics step completes
**Solution:** Queue actions for NEXT step while current step runs

```cpp
// Double-buffered action queue
struct ActionBuffer {
    AlignedVector32<float> currentActions;  // Being consumed
    AlignedVector32<float> nextActions;     // Being produced
    std::atomic<bool> swapReady;
};
```

**Expected Gain:** Hides action generation latency

---

### Phase 3: Memory Layout Optimization 📦

#### 3.1 Structure of Arrays (SoA) for Environment Data
**Problem:** AoS layout causes cache misses when accessing single field across envs
**Solution:** Convert to SoA layout

```cpp
// Before: Array of Structures
struct EnvState {
    float observations[256];
    float actions[32];
    float rewards[4];
    bool done;
};
std::vector<EnvState> envs;

// After: Structure of Arrays
struct EnvBatch {
    AlignedVector32<float> allObservations;  // [N * 256]
    AlignedVector32<float> allActions;       // [N * 32]
    AlignedVector32<float> allRewards;       // [N * 4]
    std::vector<bool> allDones;              // [N]
};
```

**Expected Gain:** 3-5x better cache utilization

#### 3.2 Cache-Line Alignment for Thread-Local Data
**Problem:** False sharing between thread-local variables
**Solution:** Align to cache line boundaries (64 bytes)

```cpp
template<typename T>
struct CacheAligned {
    alignas(64) T data;
    char padding[64 - sizeof(T) % 64];  // Pad to cache line
};
```

**Expected Gain:** 10-30% faster multi-threaded access

#### 3.3 Prefetching for Sequential Access
**Problem:** Memory latency not hidden during sequential scans
**Solution:** Software prefetching 200-300 cycles ahead

```cpp
inline void PrefetchAhead(const float* data, size_t ahead) {
    for (size_t i = 0; i < size; i += 8) {
        _mm_prefetch(data + i + ahead, _MM_HINT_T0);  // L1 prefetch
        // Process current data
        process(data + i);
    }
}
```

**Expected Gain:** 15-25% faster sequential access

---

### Phase 4: Batched Training Optimization 🎯

#### 4.1 Larger Batch Sizes with Gradient Accumulation
**Problem:** Small batches don't utilize CPU/GPU fully
**Solution:** Accumulate gradients over multiple micro-batches

```cpp
// Accumulate gradients over K steps before optimizer update
for (int accumStep = 0; accumStep < accumulationSteps; accumStep++) {
    SampleBatch(buffer, microBatch);
    ComputeGradients(microBatch, tempGrads);
    AccumulateGradients(accumulatedGrads, tempGrads);
}
// Single optimizer step with accumulated gradients
optimizer.step(accumulatedGrads);
```

**Expected Gain:** 2-4x better throughput, same convergence

#### 4.2 Prioritized Experience Replay with SIMD
**Problem:** PER sampling is sequential bottleneck
**Solution:** Vectorized priority sampling

```cpp
class VectorizedPrioritizedReplay {
    // Sum tree with SIMD traversal
    AlignedVector32<float> priorityTree;  // Power of 2 size
    
    void SampleBatch(int batchSize, int* indices) {
        // Sample multiple priorities in parallel using AVX2
        for (int i = 0; i < batchSize; i += 8) {
            __m256 priorities = SamplePrioritiesAVX2();
            _mm256_storeu_ps(tempPriorities + i, priorities);
        }
    }
};
```

**Expected Gain:** 3-5x faster sampling

#### 4.3 On-the-Fly Data Augmentation
**Problem:** Limited diversity in training data
**Solution:** Augment observations during replay (no storage cost)

```cpp
void AugmentBatch(float* observations, int batchSize, int dim) {
    #pragma omp parallel for
    for (int i = 0; i < batchSize; i++) {
        float* obs = observations + i * dim;
        // Add small noise, rotate, scale
        AddGaussianNoiseAVX2(obs, dim, 0.01f);
    }
}
```

**Expected Gain:** Better sample efficiency (indirect SPS gain)

---

### Phase 5: Latent Memory Vectorization 🧠

#### 5.1 Batched Latent State Updates
**Problem:** Latent memory updated sequentially per environment
**Solution:** Vectorized batch update across all environments

```cpp
class BatchedLatentMemory {
    // SoA layout: [latentDim][numEnvs] instead of [numEnvs][latentDim]
    AlignedVector32<float> latentPositions;  // [latentDim * numEnvs]
    AlignedVector32<float> latentVelocities;
    
    void UpdateBatch(const float* actions, const float* rewards, int numEnvs) {
        // Update all latent states in parallel
        #pragma omp simd aligned(latentPositions, latentVelocities: 32)
        for (int i = 0; i < latentDim * numEnvs; i++) {
            latentVelocities[i] = decay * latentVelocities[i] + learningRate * rewards[i];
            latentPositions[i] += latentVelocities[i];
        }
    }
};
```

**Expected Gain:** 5-10x faster latent updates

#### 5.2 Hierarchical Latent Attention
**Problem:** All environments compute attention independently
**Solution:** Share attention computation across similar states

```cpp
class AttentionLatentMemory {
    // Cluster environments by state similarity
    std::vector<int> stateClusters;
    AlignedVector32<float> clusterCentroids;
    
    void ComputeAttentionBatch(const float* states, int numEnvs) {
        // Only compute attention for cluster centroids
        // Broadcast to cluster members
        for (int cluster : uniqueClusters) {
            ComputeAttention(centroid[cluster]);
            BroadcastToCluster(cluster);
        }
    }
};
```

**Expected Gain:** 2-3x faster attention (depending on cluster count)

---

### Phase 6: Rendering Optimization 🖼️

#### 6.1 Async Rendering Pipeline
**Problem:** Rendering blocks environment stepping
**Solution:** Render in separate thread with double-buffered state

```cpp
class AsyncRenderer {
    struct RenderState {
        AlignedVector32<float> robotPositions;
        AlignedVector32<float> robotVelocities;
        std::atomic<bool> ready;
    };
    
    RenderState frontBuffer;   // Being rendered
    RenderState backBuffer;    // Being updated
    
    void RenderThread() {
        while (running) {
            if (backBuffer.ready) {
                SwapBuffers();
                RenderFrame(frontBuffer);
            }
        }
    }
};
```

**Expected Gain:** Zero rendering overhead on training loop

#### 6.2 Level-of-Detail Rendering
**Problem:** Full detail rendering for all robots
**Solution:** LOD based on camera distance

```cpp
enum class LOD {
    HIGH,    // Full mesh, all satellites (render env only)
    MEDIUM,  // Simplified mesh
    LOW      // Bounding boxes only (headless envs)
};

void RenderRobot(const Robot& robot, LOD level) {
    switch (level) {
        case LOD::HIGH: RenderFullDetail(robot); break;
        case LOD::MEDIUM: RenderSimplified(robot); break;
        case LOD::LOW: RenderBoundingBox(robot); break;
    }
}
```

**Expected Gain:** 50-80% faster rendering for headless envs

---

## Implementation Priority

| Priority | Phase | Expected SPS Gain | Complexity |
|----------|-------|-------------------|------------|
| 1 | Phase 2.1: Multi-Physics Parallelization | 4-8x | Medium |
| 2 | Phase 3.1: SoA Memory Layout | 2-3x | Medium |
| 3 | Phase 1.2: Batched GEMM with Eigen | 1.5-2x | Low |
| 4 | Phase 4.1: Larger Batches + Accumulation | 2-4x | Low |
| 5 | Phase 2.2: Lock-Free State Harvesting | 1.5-2x | Medium |
| 6 | Phase 5.1: Batched Latent Memory | 1.3-1.5x | Low |
| 7 | Phase 1.1: Layer Fusion | 1.2-1.3x | Medium |
| 8 | Phase 6.1: Async Rendering | 1.1-1.2x | High |

**Total Expected SPS Gain: 20-50x** (multiplicative when combined)

---

## Testing Strategy

Following the polyglot-test-agent methodology:

1. **Research Phase**: Profile current implementation to identify exact bottlenecks
2. **Planning Phase**: Implement optimizations one at a time with benchmarks
3. **Implementation Phase**: 
   - Write unit tests for each optimized component
   - Verify numerical correctness (comparing before/after outputs)
   - Benchmark SPS improvement
   - Build and run to verify compilation

### Test Files to Create:
- `src/OptimizationTests.cpp` - Unit tests for optimized components
- `benchmarks/sps_benchmark.cpp` - SPS comparison before/after
- `benchmarks/memory_benchmark.cpp` - Cache efficiency tests

---

## Next Steps

1. **Create optimization branch** for safe experimentation
2. **Implement Phase 2.1 first** (Multi-Physics) - highest ROI
3. **Profile after each phase** to verify improvements
4. **Create tests** using polyglot-test-agent skill for each component
5. **Document final SPS** and compare against baseline
