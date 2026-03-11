#include "common.h"
#include <cstdint>

// Simplified Layer and Filtering implementation for WASM
namespace WasmLayers {
    static constexpr JPH::ObjectLayer STATIC = 0;
    static constexpr JPH::ObjectLayer MOVING_BASE = 1;
}

namespace WasmBroadPhaseLayers {
    static constexpr JPH::BroadPhaseLayer STATIC(0);
    static constexpr JPH::BroadPhaseLayer DYNAMIC(1);
    static constexpr uint32_t NUM_LAYERS = 2;
}

class WasmBPLayerInterface final : public JPH::BroadPhaseLayerInterface {
public:
    virtual uint32_t GetNumBroadPhaseLayers() const override { return WasmBroadPhaseLayers::NUM_LAYERS; }
    virtual JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer inLayer) const override {
        if (inLayer == WasmLayers::STATIC) return WasmBroadPhaseLayers::STATIC;
        return WasmBroadPhaseLayers::DYNAMIC;
    }
};

class WasmObjectVsBroadPhaseLayerFilter final : public JPH::ObjectVsBroadPhaseLayerFilter {
public:
    virtual bool ShouldCollide(JPH::ObjectLayer inLayer1, JPH::BroadPhaseLayer inLayer2) const override {
        if (inLayer1 == WasmLayers::STATIC) return inLayer2 == WasmBroadPhaseLayers::DYNAMIC;
        return true;
    }
};

class WasmObjectLayerPairFilter final : public JPH::ObjectLayerPairFilter {
public:
    virtual bool ShouldCollide(JPH::ObjectLayer inObject1, JPH::ObjectLayer inObject2) const override {
        if (inObject1 == WasmLayers::STATIC && inObject2 == WasmLayers::STATIC) return false;
        if (inObject1 == WasmLayers::STATIC || inObject2 == WasmLayers::STATIC) return true;
        return inObject1 == inObject2; // Dimensional Ghosting: only collide if in same env
    }
};

extern "C" {

EMSCRIPTEN_KEEPALIVE
PhysicsWorld* create_physics_world(uint32_t numEnvs) {
    PhysicsWorld* world = new PhysicsWorld();
    world->numEnvs = numEnvs;
    
    // Initialize Jolt
    JPH::RegisterDefaultAllocator();
    JPH::Factory::sInstance = new JPH::Factory();
    
    world->tempAllocator = new JPH::TempAllocatorImpl(10 * 1024 * 1024);
    world->jobSystem = new JPH::JobSystemThreadPool(JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, 1);
    
    world->bpInterface = new WasmBPLayerInterface();
    world->objVsBpFilter = new WasmObjectVsBroadPhaseLayerFilter();
    world->objPairFilter = new WasmObjectLayerPairFilter();
    
    world->physicsSystem = new JPH::PhysicsSystem();
    world->physicsSystem->Init(
        1024 * numEnvs, // Max Bodies
        0,              // Num Body Mutexes
        1024 * numEnvs, // Max Body Pairs
        1024 * numEnvs, // Max Contact Constraints
        *world->bpInterface,
        *world->objVsBpFilter,
        *world->objPairFilter
    );
    
    return world;
}

EMSCRIPTEN_KEEPALIVE
void destroy_physics_world(PhysicsWorld* world) {
    if (world) {
        delete world->physicsSystem;
        delete world->bpInterface;
        delete world->objVsBpFilter;
        delete world->objPairFilter;
        delete world->jobSystem;
        delete world->tempAllocator;
        delete world;
    }
}

EMSCRIPTEN_KEEPALIVE
void physics_step(PhysicsWorld* world, float deltaTime) {
    if (world && world->physicsSystem) {
        world->physicsSystem->Update(deltaTime, 1, world->tempAllocator, world->jobSystem);
    }
}

} // extern "C"
