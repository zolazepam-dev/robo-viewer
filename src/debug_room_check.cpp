// Test to check if rooms are created with static bodies
#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <iostream>
#include "VectorizedEnv.h"

int main() {
    VectorizedEnv vecEnv(2);
    vecEnv.Init();
    
    auto* physicsSystem = vecEnv.GetGlobalPhysics();
    JPH::BodyIDVector bodies;
    physicsSystem->GetBodies(bodies);
    
    std::cout << "Total bodies: " << bodies.size() << std::endl;
    
    int staticCount = 0;
    int dynamicCount = 0;
    
    JPH::BodyInterface& bodyInterface = physicsSystem->GetBodyInterface();
    for (const JPH::BodyID& bodyId : bodies) {
        if (bodyId.IsInvalid()) continue;
        
        JPH::EMotionType motionType = bodyInterface.GetMotionType(bodyId);
        JPH::ObjectLayer layer = bodyInterface.GetObjectLayer(bodyId);
        
        if (motionType == JPH::EMotionType::Static) {
            staticCount++;
            JPH::RVec3 pos = bodyInterface.GetPosition(bodyId);
            std::cout << "Static body at (" << pos.GetX() << ", " << pos.GetY() << ", " << pos.GetZ() << ")" << std::endl;
        } else {
            dynamicCount++;
            JPH::RVec3 pos = bodyInterface.GetPosition(bodyId);
            std::cout << "Dynamic body at (" << pos.GetX() << ", " << pos.GetY() << ", " << pos.GetZ() << ")" << std::endl;
        }
    }
    
    std::cout << "Static bodies: " << staticCount << std::endl;
    std::cout << "Dynamic bodies: " << dynamicCount << std::endl;
    
    return 0;
}
