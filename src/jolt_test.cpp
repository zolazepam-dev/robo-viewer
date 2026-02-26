#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayer.h>
#include <Jolt/Physics/Collision/ObjectLayer.h>

#include <iostream>

namespace Layers
{
    static constexpr JPH::ObjectLayer STATIC = 0;
    static constexpr JPH::ObjectLayer MOVING = 1;
}

namespace BroadPhaseLayers
{
    static constexpr JPH::BroadPhaseLayer STATIC(0);
    static constexpr JPH::BroadPhaseLayer MOVING(1);
    static constexpr uint NUM_LAYERS = 2;
}

class BPLayerInterfaceImpl final : public JPH::BroadPhaseLayerInterface
{
public:
    virtual uint GetNumBroadPhaseLayers() const override { return BroadPhaseLayers::NUM_LAYERS; }

    virtual JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer inLayer) const override
    {
        if (inLayer == Layers::STATIC) return BroadPhaseLayers::STATIC;
        return BroadPhaseLayers::MOVING;
    }
};

class ObjectVsBroadPhaseLayerFilterImpl final : public JPH::ObjectVsBroadPhaseLayerFilter
{
public:
    virtual bool ShouldCollide(JPH::ObjectLayer inLayer1, JPH::BroadPhaseLayer inLayer2) const override
    {
        if (inLayer1 == Layers::STATIC) return inLayer2 == BroadPhaseLayers::MOVING;
        return true;
    }
};

class ObjectLayerPairFilterImpl final : public JPH::ObjectLayerPairFilter
{
public:
    virtual bool ShouldCollide(JPH::ObjectLayer inObject1, JPH::ObjectLayer inObject2) const override
    {
        if (inObject1 == Layers::STATIC && inObject2 == Layers::STATIC) return false;
        if (inObject1 == Layers::STATIC || inObject2 == Layers::STATIC) return true;
        return true;
    }
};

int main() {
    std::cout << "[JOLT TEST] Starting Jolt initialization test..." << std::endl;
    
    // Register Jolt types
    JPH::RegisterDefaultAllocator();
    JPH::Factory::sInstance = new JPH::Factory();
    JPH::RegisterTypes();
    
    std::cout << "[JOLT TEST] Jolt types registered" << std::endl;
    
    // Create temp allocator
    JPH::TempAllocatorImpl temp_allocator(10 * 1024 * 1024);
    std::cout << "[JOLT TEST] Temp allocator created" << std::endl;
    
    // Create job system
    JPH::JobSystemThreadPool job_system(JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, 4);
    std::cout << "[JOLT TEST] Job system created" << std::endl;
    
    // Create layer interfaces
    BPLayerInterfaceImpl bp_layer_interface;
    ObjectVsBroadPhaseLayerFilterImpl object_vs_bp_filter;
    ObjectLayerPairFilterImpl object_pair_filter;
    
    std::cout << "[JOLT TEST] Layer interfaces created" << std::endl;
    
    // Create physics system
    JPH::PhysicsSystem physics_system;
    physics_system.Init(1024, 0, 1024, 1024, bp_layer_interface, object_vs_bp_filter, object_pair_filter);
    std::cout << "[JOLT TEST] Physics system initialized" << std::endl;
    
    std::cout << "[JOLT TEST] Done." << std::endl;
    
    // Cleanup
    delete JPH::Factory::sInstance;
    JPH::Factory::sInstance = nullptr;
    
    return 0;
}