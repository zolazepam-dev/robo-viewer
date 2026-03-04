/**
 * @file PhysicsCore.h
 * @brief High-performance Jolt Physics engine wrapper for JOLTrl
 * 
 * This file contains the PhysicsCore class, which manages the Jolt Physics system
 * with optimizations specifically tailored for reinforcement learning applications.
 * Key features include:
 * - Dimensional Ghosting for parallel environment simulation
 * - Thread pinning for maximum CPU utilization
 * - Zero-allocation physics loop for high throughput
 * - Optimized collision filtering for 1v1 robot combat
 */

#pragma once

// STRICT REQUIREMENT: Jolt.h must be included first
#include <Jolt/Jolt.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayer.h>
#include <Jolt/Physics/Collision/ObjectLayer.h>
#include <Jolt/Core/JobSystemThreadPool.h>

#include <vector>

/**
 * @namespace Layers
 * @brief Defines physics object layers for collision filtering
 * 
 * Layers are used to implement the Dimensional Ghosting technique, allowing
 * multiple environments to coexist in the same physics world without collisions.
 */
namespace Layers
{
    // Layer 0 is the universal static layer (e.g., the floor)
    static constexpr JPH::ObjectLayer STATIC = 0;

    // Every parallel environment gets its own exclusive layer starting from 1
    // e.g., Env 0 -> Layer 1, Env 1 -> Layer 2.
    static constexpr JPH::ObjectLayer MOVING_BASE = 1;

    // Ghost layers for visuals that should NOT collide with anything
    // Offset by a large enough number to not overlap with MOVING_BASE
    static constexpr JPH::ObjectLayer GHOST_BASE = 5000;
}

/**
 * @namespace BroadPhaseLayers
 * @brief Defines broad-phase collision layers
 * 
 * Maps object layers to broad-phase trees for optimized collision detection
 */
namespace BroadPhaseLayers
{
    static constexpr JPH::BroadPhaseLayer STATIC(0);
    static constexpr JPH::BroadPhaseLayer DYNAMIC(1);
    static constexpr uint NUM_LAYERS = 2;
}

/**
 * @class BPLayerInterfaceImpl
 * @brief Maps object layers to broad-phase layers
 * 
 * Converts thousands of object layers (one per environment) into just 2 broad-phase
 * layers (Static and Dynamic) for efficient collision detection.
 */
class BPLayerInterfaceImpl final : public JPH::BroadPhaseLayerInterface
{
public:
    /**
     * @brief Constructor
     * @param numEnvs Number of parallel environments
     */
    BPLayerInterfaceImpl(uint32_t numEnvs) : mNumEnvs(numEnvs)
    {
    }

    /** @brief Get number of broad-phase layers */
    virtual uint GetNumBroadPhaseLayers() const override { return BroadPhaseLayers::NUM_LAYERS; }

    /**
     * @brief Convert object layer to broad-phase layer
     * @param inLayer Object layer to convert
     * @return Corresponding broad-phase layer
     */
    virtual JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer inLayer) const override
    {
        if (inLayer == Layers::STATIC) return BroadPhaseLayers::STATIC;
        return BroadPhaseLayers::DYNAMIC;
    }

#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    /**
     * @brief Get name of broad-phase layer for profiling
     * @param inLayer Broad-phase layer
     * @return Name of the layer
     */
    virtual const char* GetBroadPhaseLayerName(JPH::BroadPhaseLayer inLayer) const override
    {
        switch ((JPH::BroadPhaseLayer::Type)inLayer)
        {
        case (JPH::BroadPhaseLayer::Type)BroadPhaseLayers::STATIC: return "Static";
        case (JPH::BroadPhaseLayer::Type)BroadPhaseLayers::DYNAMIC: return "Dynamic";
        default: return "Invalid";
        }
    }
#endif // JPH_EXTERNAL_PROFILE || JPH_PROFILE_ENABLED

private:
    uint32_t mNumEnvs;  ///< Number of parallel environments
};

/**
 * @class ObjectVsBroadPhaseLayerFilterImpl
 * @brief Filter for object vs broad-phase layer collisions
 */
class ObjectVsBroadPhaseLayerFilterImpl final : public JPH::ObjectVsBroadPhaseLayerFilter
{
public:
    /**
     * @brief Determine if an object layer should collide with a broad-phase layer
     * @param inLayer1 Object layer
     * @param inLayer2 Broad-phase layer
     * @return True if they should collide
     */
    virtual bool ShouldCollide(JPH::ObjectLayer inLayer1, JPH::BroadPhaseLayer inLayer2) const override
    {
        if (inLayer1 == Layers::STATIC) return inLayer2 == BroadPhaseLayers::DYNAMIC;
        return true; // Dynamic objects can collide with both Static and Dynamic broadphases
    }
};

/**
 * @class ObjectLayerPairFilterImpl
 * @brief Core Dimensional Ghosting collision filter logic
 * 
 * Prevents robots from different environments from colliding with each other,
 * while allowing collisions within the same environment.
 */
class ObjectLayerPairFilterImpl final : public JPH::ObjectLayerPairFilter
{
public:
    /**
     * @brief Determine if two object layers should collide
     * @param inObject1 First object layer
     * @param inObject2 Second object layer
     * @return True if they should collide
     */
    virtual bool ShouldCollide(JPH::ObjectLayer inObject1, JPH::ObjectLayer inObject2) const override
    {
        // 1. Ghost layers NEVER collide with anything
        if (inObject1 >= Layers::GHOST_BASE || inObject2 >= Layers::GHOST_BASE) return false;

        // 2. Static objects don't collide with other static objects
        if (inObject1 == Layers::STATIC && inObject2 == Layers::STATIC) return false;

        // 3. Everything collides with the static environment (Floor)
        if (inObject1 == Layers::STATIC || inObject2 == Layers::STATIC) return true;

        // 4. Two dynamic objects ONLY collide if they belong to the exact same environment
        return inObject1 == inObject2;
    }
};

/**
 * @class PhysicsCore
 * @brief High-performance Jolt Physics system manager
 * 
 * Manages the entire physics simulation with optimizations for reinforcement learning:
 * - Thread pinning for maximum CPU utilization
 * - Zero-allocation physics loop
 * - Parallel environment support via Dimensional Ghosting
 * - Optimized memory pooling
 */
class PhysicsCore
{
public:
    PhysicsCore() = default;
    ~PhysicsCore();

    PhysicsCore(const PhysicsCore&) = delete;
    PhysicsCore& operator=(const PhysicsCore&) = delete;
    PhysicsCore(PhysicsCore&&) = delete;
    PhysicsCore& operator=(PhysicsCore&&) = delete;

    /**
     * @brief Initialize the physics system
     * @param numParallelEnvs Number of parallel environments to support
     * @return True if initialization succeeded
     */
    bool Init(uint32_t numParallelEnvs);
    
    /**
     * @brief Step the physics simulation
     * @param deltaTime Time to simulate in seconds
     */
    void Step(float deltaTime);
    
    /** @brief Shutdown the physics system */
    void Shutdown();

    /** @brief Get reference to the physics system */
    JPH::PhysicsSystem& GetPhysicsSystem() { return *mPhysicsSystem; }
    const JPH::PhysicsSystem& GetPhysicsSystem() const { return *mPhysicsSystem; }
    
    /** @brief Get pointer to temp allocator */
    JPH::TempAllocator* GetTempAllocator() { return mTempAllocator; }
    
    /** @brief Get pointer to job system */
    JPH::JobSystem* GetJobSystem() { return mJobSystem; }
    
    /**
     * @brief Set physics settings
     * @param settings Physics settings to apply
     */
    void SetSettings(const JPH::PhysicsSettings& settings) {
        if (mPhysicsSystem) mPhysicsSystem->SetPhysicsSettings(settings);
    }
    
    /**
     * @brief Get all body IDs in the physics system that belong to specific object layers
     * @param outBodies Output vector to store matching body IDs
     * @param layers Vector of object layers to include in the result
     */
    void GetBodiesByLayers(JPH::BodyIDVector& outBodies, const std::vector<JPH::ObjectLayer>& layers) const;

    /**
     * @brief Get all body IDs in the physics system that belong to a specific object layer
     * @param outBodies Output vector to store matching body IDs
     * @param layer Object layer to filter by
     */
    void GetBodiesByLayer(JPH::BodyIDVector& outBodies, JPH::ObjectLayer layer) const;

    /** @brief Get current physics settings */
    JPH::PhysicsSettings GetSettings() const {
        return mPhysicsSystem ? mPhysicsSystem->GetPhysicsSettings() : JPH::PhysicsSettings();
    }

private:
    JPH::TempAllocatorImpl* mTempAllocator = nullptr;                ///< Temporary memory allocator
    JPH::JobSystemThreadPool* mJobSystem = nullptr;                  ///< Thread pool for physics jobs
    BPLayerInterfaceImpl* mBroadPhaseLayerInterface = nullptr;       ///< Broad-phase layer interface
    ObjectVsBroadPhaseLayerFilterImpl* mObjectVsBroadPhaseLayerFilter = nullptr; ///< Object vs broad-phase filter
    ObjectLayerPairFilterImpl* mObjectLayerPairFilter = nullptr;     ///< Object vs object filter
    JPH::PhysicsSystem* mPhysicsSystem = nullptr;                    ///< Main physics system

    bool mInitialized = false;  ///< Initialization state
    uint32_t mNumEnvs = 1;      ///< Number of parallel environments
};
