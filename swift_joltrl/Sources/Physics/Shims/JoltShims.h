#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayer.h>
#include <Jolt/Physics/Collision/ObjectLayer.h>
#include <Jolt/Physics/PhysicsSystem.h>

namespace SwiftJolt {

    // A simple BroadPhaseLayerInterface implementation that Swift can use
    class SimpleBroadPhaseLayerInterface : public JPH::BroadPhaseLayerInterface {
    public:
        SimpleBroadPhaseLayerInterface() {
            mObjectToBroadPhase[0] = JPH::BroadPhaseLayer(0); // STATIC
            mObjectToBroadPhase[1] = JPH::BroadPhaseLayer(1); // DYNAMIC
        }

        virtual uint GetNumBroadPhaseLayers() const override { return 2; }
        virtual JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer inLayer) const override {
            return mObjectToBroadPhase[inLayer];
        }

    private:
        JPH::BroadPhaseLayer mObjectToBroadPhase[2];
    };

    // A simple ObjectVsBroadPhaseLayerFilter implementation
    class SimpleObjectVsBroadPhaseLayerFilter : public JPH::ObjectVsBroadPhaseLayerFilter {
    public:
        virtual bool ShouldCollide(JPH::ObjectLayer inLayer1, JPH::BroadPhaseLayer inLayer2) const override {
            switch (inLayer1) {
                case 0: return inLayer2 == JPH::BroadPhaseLayer(1); // Static only collides with Dynamic
                case 1: return true; // Dynamic collides with everything
                default: return false;
            }
        }
    };

    // A simple ObjectLayerPairFilter implementation
    class SimpleObjectLayerPairFilter : public JPH::ObjectLayerPairFilter {
    public:
        virtual bool ShouldCollide(JPH::ObjectLayer inObject1, JPH::ObjectLayer inObject2) const override {
            switch (inObject1) {
                case 0: return inObject2 == 1; // Static only collides with Dynamic
                case 1: return true; // Dynamic collides with everything
                default: return false;
            }
        }
    };

    // Helper functions to create these objects for Swift
    inline JPH::BroadPhaseLayerInterface* CreateBroadPhaseLayerInterface() { return new SimpleBroadPhaseLayerInterface(); }
    inline JPH::ObjectVsBroadPhaseLayerFilter* CreateObjectVsBroadPhaseLayerFilter() { return new SimpleObjectVsBroadPhaseLayerFilter(); }
    inline JPH::ObjectLayerPairFilter* CreateObjectLayerPairFilter() { return new SimpleObjectLayerPairFilter(); }

} // namespace SwiftJolt
