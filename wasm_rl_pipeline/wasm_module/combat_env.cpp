#include "common.h"
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>

extern "C" {

EMSCRIPTEN_KEEPALIVE
CombatEnvState* create_combat_env(PhysicsWorld* world) {
    if (!world || !world->physicsSystem) return nullptr;
    
    CombatEnvState* env = new CombatEnvState();
    env->physicsWorld = world;
    
    JPH::BodyInterface& bodyInterface = world->physicsSystem->GetBodyInterface();
    
    // 1. Create Floor (Object Layer 0 - Static)
    // We only create one floor per physics world usually, 
    // but for simplicity in this WASM module, each env can have its own "sector"
    // or we assume they share a massive one. 
    // Let's create a floor if this is the first environment or if we want per-env floors.
    // For Dimensional Ghosting, we just need to ensure the floor is in Layer 0.
    
    // 2. Create Robots in their specific environment layer (env->id + 1)
    // We'll set the ID later in vectorized_env, but let's assume it's set or use a default
    
    // Define a simple box shape for robots for now
    JPH::BoxShapeSettings boxSettings(JPH::Vec3(0.5f, 0.5f, 0.5f));
    JPH::Shape::ShapeResult shapeResult = boxSettings.Create();
    JPH::Ref<JPH::Shape> robotShape = shapeResult.Get();
    
    for (int i = 0; i < 2; i++) {
        float startX = (i == 0) ? -2.0f : 2.0f;
        JPH::BodyCreationSettings settings(robotShape, JPH::RVec3(startX, 1.0f, 0.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, 1); // Layer 1 default
        
        JPH::Body* body = bodyInterface.CreateBody(settings);
        bodyInterface.AddBody(body->GetID(), JPH::EActivation::Activate);
        env->physicsBodyIds[i] = body->GetID();
        
        // Initialize state
        env->robots[i].x = startX;
        env->robots[i].y = 1.0f;
        env->robots[i].z = 0.0f;
        env->robots[i].health = 100;
        env->robots[i].active = 1;
    }
    
    env->arenaBounds[0] = -18.0f; env->arenaBounds[1] = 18.0f;
    env->arenaBounds[2] = 0.0f;   env->arenaBounds[3] = 10.0f;
    env->arenaBounds[4] = -18.0f; env->arenaBounds[5] = 18.0f;
    
    env->stepCount = 0;
    env->done = 0;
    env->reward[0] = 0.0f;
    env->reward[1] = 0.0f;
    
    return env;
}

EMSCRIPTEN_KEEPALIVE
void combat_env_step(CombatEnvState* env, const float* actions, int actionDim) {
    if (!env || env->done || !env->physicsWorld) return;
    
    JPH::BodyInterface& bodyInterface = env->physicsWorld->physicsSystem->GetBodyInterface();
    
    // Apply actions as forces/impulses
    for (int i = 0; i < 2; i++) {
        float moveX = actions[i * actionDim + 0];
        float moveZ = actions[i * actionDim + 1];
        
        bodyInterface.AddImpulse(env->physicsBodyIds[i], JPH::Vec3(moveX * 10.0f, 0, moveZ * 10.0f));
    }
    
    // Sync Jolt state back to RobotState for transport to JS
    for (int i = 0; i < 2; i++) {
        JPH::RVec3 pos = bodyInterface.GetPosition(env->physicsBodyIds[i]);
        JPH::Quat rot = bodyInterface.GetRotation(env->physicsBodyIds[i]);
        JPH::Vec3 vel = bodyInterface.GetLinearVelocity(env->physicsBodyIds[i]);
        
        env->robots[i].x = pos.GetX();
        env->robots[i].y = pos.GetY();
        env->robots[i].z = pos.GetZ();
        env->robots[i].vx = vel.GetX();
        env->robots[i].vy = vel.GetY();
        env->robots[i].vz = vel.GetZ();
        env->robots[i].rotation[0] = rot.GetW();
        env->robots[i].rotation[1] = rot.GetX();
        env->robots[i].rotation[2] = rot.GetY();
        env->robots[i].rotation[3] = rot.GetZ();
        
        // Simple OOB check
        if (pos.GetY() < -1.0f) env->done = 1;
    }
    
    env->stepCount++;
    if (env->stepCount >= 1000) env->done = 1;
}

EMSCRIPTEN_KEEPALIVE
void reset_env(CombatEnvState* env) {
    if (!env || !env->physicsWorld) return;
    
    JPH::BodyInterface& bodyInterface = env->physicsWorld->physicsSystem->GetBodyInterface();
    
    for (int i = 0; i < 2; i++) {
        float startX = (i == 0) ? -2.0f : 2.0f;
        bodyInterface.SetPositionAndRotation(env->physicsBodyIds[i], JPH::RVec3(startX, 1.0f, 0.0f), JPH::Quat::sIdentity(), JPH::EActivation::Activate);
        bodyInterface.SetLinearAndAngularVelocity(env->physicsBodyIds[i], JPH::Vec3::sZero(), JPH::Vec3::sZero());
        
        env->robots[i].health = 100;
    }
    env->stepCount = 0;
    env->done = 0;
    env->reward[0] = 0.0f;
    env->reward[1] = 0.0f;
}

EMSCRIPTEN_KEEPALIVE
void get_observation(const CombatEnvState* env, float* obs, int obsDim) {
    if (!env) return;
    for (int i = 0; i < 10 && i < obsDim; i++) {
        obs[i] = env->robots[0].x; // Simple mock obs
    }
}

} // extern "C"
