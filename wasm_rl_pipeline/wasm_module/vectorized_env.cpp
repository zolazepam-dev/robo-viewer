#include "common.h"

// External C functions from combat_env.cpp
extern "C" {
    CombatEnvState* create_combat_env(PhysicsWorld* world);
    void combat_env_step(CombatEnvState* env, const float* actions, int actionDim);
    void reset_env(CombatEnvState* env);
    void get_observation(const CombatEnvState* env, float* obs, int obsDim);
}

class VectorizedEnvironment {
private:
    std::vector<CombatEnvState*> environments;
    PhysicsWorld* mWorld;
    int observationDim;
    int actionDim;
    
public:
    VectorizedEnvironment(PhysicsWorld* world, int numEnvs, int obsDim = 256, int actDim = 56)
        : mWorld(world), observationDim(obsDim), actionDim(actDim) {
        
        // Create a common ground plane in the physics world
        JPH::BodyInterface& bodyInterface = world->physicsSystem->GetBodyInterface();
        JPH::BoxShapeSettings floorSettings(JPH::Vec3(100.0f, 1.0f, 100.0f));
        JPH::Shape::ShapeResult floorResult = floorSettings.Create();
        JPH::BodyCreationSettings floorCreationSettings(floorResult.Get(), JPH::RVec3(0, -1.0f, 0), JPH::Quat::sIdentity(), JPH::EMotionType::Static, 0); // Layer 0
        bodyInterface.CreateAndAddBody(floorCreationSettings, JPH::EActivation::DontActivate);

        for (int i = 0; i < numEnvs; i++) {
            CombatEnvState* env = create_combat_env(world);
            env->id = i;
            
            // Assign exclusive layer for Dimensional Ghosting (Layer 1, 2, 3...)
            JPH::ObjectLayer envLayer = (JPH::ObjectLayer)(i + 1);
            bodyInterface.SetObjectLayer(env->physicsBodyIds[0], envLayer);
            bodyInterface.SetObjectLayer(env->physicsBodyIds[1], envLayer);
            
            environments.push_back(env);
        }
    }
    
    ~VectorizedEnvironment() {
        for (auto env : environments) {
            delete env;
        }
    }
    
    void step(const float* actions, int batchSize) {
        for (int i = 0; i < batchSize; i++) {
            if (i < environments.size()) {
                combat_env_step(environments[i], &actions[i * actionDim], actionDim);
            }
        }
    }
    
    void reset() {
        for (auto env : environments) {
            reset_env(env);
        }
    }
    
    void getObservations(float* obs) {
        for (int i = 0; i < environments.size(); i++) {
            get_observation(environments[i], &obs[i * observationDim], observationDim);
        }
    }
    
    void getRewards(float* rewards) {
        for (int i = 0; i < environments.size(); i++) {
            rewards[i] = environments[i]->reward[0]; // Agent 0 reward
        }
    }
    
    void getDones(int* dones) {
        for (int i = 0; i < environments.size(); i++) {
            dones[i] = environments[i]->done;
        }
    }
    
    int getNumEnvironments() const { return (int)environments.size(); }
};
