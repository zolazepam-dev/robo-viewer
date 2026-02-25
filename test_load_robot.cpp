#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/JobSystemThreadPool.h>

#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main() {
    // Initialize Jolt
    JPH::RegisterDefaultAllocator();
    JPH::JPHInit();
    JPH::RegisterTypes();
    
    // Create physics system
    JPH::PhysicsSystem physicsSystem;
    physicsSystem.Init(JPH::PhysicsSettings());
    
    std::cout << "Physics system initialized" << std::endl;
    
    // Create job system
    JPH::JobSystemThreadPool jobSystem(10);
    
    // Create body interface
    JPH::BodyInterface& bodyInterface = physicsSystem.GetBodyInterface();
    
    std::cout << "Loading robot config..." << std::endl;
    std::ifstream file("robots/combat_bot.json");
    if (!file.is_open()) {
        std::cerr << "Failed to open config file" << std::endl;
        return 1;
    }
    
    json config;
    file >> config;
    file.close();
    
    std::cout << "Config loaded" << std::endl;
    
    // Test core body creation
    const float coreRadius = config["core"].value("radius", 0.5f);
    const float coreMass = config["core"].value("mass", 13.0f);
    
    std::cout << "Core radius: " << coreRadius << ", mass: " << coreMass << std::endl;
    
    JPH::SphereShapeSettings coreShapeSettings(coreRadius);
    coreShapeSettings.SetDensity(coreMass / (4.0f / 3.0f * 3.14159f * coreRadius * coreRadius * coreRadius));
    JPH::RefConst<JPH::Shape> coreShape = coreShapeSettings.Create().Get();
    
    JPH::BodyCreationSettings coreSettings(
        coreShape,
        JPH::RVec3(0, 1, 0),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Dynamic,
        0 // object layer
    );
    
    coreSettings.mFriction = 0.0f;
    coreSettings.mRestitution = 0.8f;
    coreSettings.mLinearDamping = 0.05f;
    coreSettings.mAngularDamping = 0.05f;
    
    JPH::Body* coreBody = bodyInterface.CreateBody(coreSettings);
    std::cout << "Core body created, ID: " << coreBody->GetID().GetIndex() << std::endl;
    
    bodyInterface.AddBody(coreBody->GetID(), JPH::EActivation::Activate);
    std::cout << "Core body added to physics system" << std::endl;
    
    // Test satellites
    const auto& satellitesConfig = config["satellites"];
    std::cout << "Number of satellites in config: " << satellitesConfig.size() << std::endl;
    
    for (int i = 0; i < satellitesConfig.size(); ++i) {
        std::cout << "Processing satellite " << i << ": ";
        try {
            auto& sat = satellitesConfig[i];
            std::cout << "ID=" << sat.value("id", 0) 
                      << ", Angle=" << sat.value("offset_angle", 0.0f)
                      << ", Elevation=" << sat.value("elevation", 0.0f);
        } catch (const std::exception& e) {
            std::cout << "ERROR: " << e.what();
        }
        std::cout << std::endl;
    }
    
    return 0;
}
