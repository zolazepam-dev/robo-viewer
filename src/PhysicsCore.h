#pragma once

// STRICT REQUIREMENT: Jolt.h must be included first
#include <Jolt/Jolt.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayer.h>
#include <Jolt/Physics/Collision/ObjectLayer.h>
#include <Jolt/Core/JobSystemThreadPool.h>

#include <vector>

namespace Layers
{
    // Layer 0 is the universal static layer (e.g., the floor)
    static constexpr JPH::ObjectLayer STATIC = 0;

    // Every parallel environment gets its own exclusive layer starting from 1
    // e.g., Env 0 -> Layer 1, Env 1 -> Layer 2.
    static constexpr JPH::ObjectLayer MOVING_BASE = 1;
}

namespace BroadPhaseLayers
{
    static constexpr JPH::BroadPhaseLayer STATIC(0);
    static constexpr JPH::BroadPhaseLayer DYNAMIC(1);
    static constexpr uint NUM_LAYERS = 2;
}

// Maps the thousands of ObjectLayers down to just 2 BroadPhase trees (Static vs Dynamic)
class BPLayerInterfaceImpl final : public JPH::BroadPhaseLayerInterface
{
public:
    BPLayerInterfaceImpl(uint32_t numEnvs) : mNumEnvs(numEnvs)
    {
    }

    virtual uint GetNumBroadPhaseLayers() const override { return BroadPhaseLayers::NUM_LAYERS; }

    virtual JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer inLayer) const override
    {
        if (inLayer == Layers::STATIC) return BroadPhaseLayers::STATIC;
        return BroadPhaseLayers::DYNAMIC;
    }

private:
    uint32_t mNumEnvs;
};

class ObjectVsBroadPhaseLayerFilterImpl final : public JPH::ObjectVsBroadPhaseLayerFilter
{
public:
    virtual bool ShouldCollide(JPH::ObjectLayer inLayer1, JPH::BroadPhaseLayer inLayer2) const override
    {
        if (inLayer1 == Layers::STATIC) return inLayer2 == BroadPhaseLayers::DYNAMIC;
        return true; // Dynamic objects can collide with both Static and Dynamic broadphases
    }
};

// DIMENSIONAL GHOSTING CORE LOGIC
class ObjectLayerPairFilterImpl final : public JPH::ObjectLayerPairFilter
{
public:
    virtual bool ShouldCollide(JPH::ObjectLayer inObject1, JPH::ObjectLayer inObject2) const override
    {
        // 1. Static objects don't collide with other static objects
        if (inObject1 == Layers::STATIC && inObject2 == Layers::STATIC) return false;
 // 2. Everything collides with the static environment (Floor)
        if (inObject1 == Layers::STATIC || inObject2 == Layers::STATIC) return true;

        // 3. Two dynamic objects ONLY collide if they belong to the exact same environment
        return inObject1 == inObject2;
    }
};

class PhysicsCore
{
public:
    PhysicsCore() = default;
    ~PhysicsCore();

    PhysicsCore(const PhysicsCore&) = delete;
    PhysicsCore& operator=(const PhysicsCore&) = delete;
    PhysicsCore(PhysicsCore&&) = delete;
    PhysicsCore& operator=(PhysicsCore&&) = delete;

    // We now pass the number of parallel environments to scale the memory pools
    bool Init(uint32_t numParallelEnvs);
    void Step(float deltaTime);
    void Shutdown();

    JPH::PhysicsSystem& GetPhysicsSystem() { return *mPhysicsSystem; }
    const JPH::PhysicsSystem& GetPhysicsSystem() const { return *mPhysicsSystem; }

private:
    JPH::TempAllocatorImpl* mTempAllocator = nullptr;
    JPH::JobSystemThreadPool* mJobSystem = nullptr;
    BPLayerInterfaceImpl* mBroadPhaseLayerInterface = nullptr;
    ObjectVsBroadPhaseLayerFilterImpl* mObjectVsBroadPhaseLayerFilter = nullptr;
    ObjectLayerPairFilterImpl* mObjectLayerPairFilter = nullptr;
    JPH::PhysicsSystem* mPhysicsSystem = nullptr;

    bool mInitialized = false;
    uint32_t mNumEnvs = 1;
};