/**
 * @file PhysicsWorld.h
 * @brief Jolt Physics world manager with RAII
 */

#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayer.h>
#include <Jolt/Physics/Collision/ObjectLayer.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/Collision/ContactListener.h>

#include <vector>
#include <memory>
#include <cstdint>
#include <array>
#include "../common/Types.h"

struct ForceSensorReading {
    std::vector<float> impulseMagnitude;
    std::vector<float> jointStress;
    void Reset(int numSatellites) {
        impulseMagnitude.assign(numSatellites, 0.0f);
        jointStress.assign(numSatellites, 0.0f);
    }
};

class ContactListener : public JPH::ContactListener {
public:
    ContactListener() = default;
    void OnContactAdded(const JPH::Body&, const JPH::Body&, const JPH::ContactManifold&, JPH::ContactSettings&) override {}
    void OnContactPersisted(const JPH::Body&, const JPH::Body&, const JPH::ContactManifold&, JPH::ContactSettings&) override {}
    void OnContactRemoved(const JPH::SubShapeIDPair&) override {}
    void Init(uint32_t numEnvs, int numSatellites);
    ForceSensorReading& GetForceReading(uint32_t envIdx, int robotIdx);
    void ResetForceReadings(uint32_t envIdx, int numSatellites);
private:
    std::vector<std::array<ForceSensorReading, 2>> mForceReadingsPerEnv;
};

class BroadPhaseLayerInterfaceImpl final : public JPH::BroadPhaseLayerInterface {
public:
    BroadPhaseLayerInterfaceImpl(uint32_t numEnvs);
    uint GetNumBroadPhaseLayers() const override;
    JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer inLayer) const override;
private:
    uint32_t mNumEnvs;
};

class ObjectVsBroadPhaseLayerFilterImpl final : public JPH::ObjectVsBroadPhaseLayerFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer inLayer1, JPH::BroadPhaseLayer inLayer2) const override;
};

class ObjectLayerPairFilterImpl final : public JPH::ObjectLayerPairFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer inObject1, JPH::ObjectLayer inObject2) const override;
};

class PhysicsWorld {
public:
    PhysicsWorld() = default;
    ~PhysicsWorld();
    PhysicsWorld(const PhysicsWorld&) = delete;
    PhysicsWorld& operator=(const PhysicsWorld&) = delete;
    
    bool Init(uint32_t numParallelEnvs, uint32_t numWorkerThreads = 10, const int* pinToCores = nullptr);
    void Shutdown();
    void Step(float deltaTime);
    
    JPH::PhysicsSystem& GetSystem() { return *mPhysicsSystem; }
    const JPH::PhysicsSystem& GetSystem() const { return *mPhysicsSystem; }
    JPH::TempAllocator* GetTempAllocator() { return mTempAllocator; }
    JPH::JobSystem* GetJobSystem() { return mJobSystem; }
    ContactListener& GetContactListener() { return *mContactListener; }
    const ContactListener& GetContactListener() const { return *mContactListener; }
    
    uint32_t GetNumEnvs() const { return mNumEnvs; }
    bool IsInitialized() const { return mInitialized; }
    
    void GetBodiesByLayers(JPH::BodyIDVector& outBodies, const std::vector<JPH::ObjectLayer>& layers) const;
    void GetBodiesByLayer(JPH::BodyIDVector& outBodies, JPH::ObjectLayer layer) const;

private:
    JPH::TempAllocatorImpl* mTempAllocator = nullptr;
    JPH::JobSystemThreadPool* mJobSystem = nullptr;
    BroadPhaseLayerInterfaceImpl* mBroadPhaseLayerInterface = nullptr;
    ObjectVsBroadPhaseLayerFilterImpl* mObjectVsBroadPhaseLayerFilter = nullptr;
    ObjectLayerPairFilterImpl* mObjectLayerPairFilter = nullptr;
    JPH::PhysicsSystem* mPhysicsSystem = nullptr;
    ContactListener* mContactListener = nullptr;
    bool mInitialized = false;
    uint32_t mNumEnvs = 1;
};
