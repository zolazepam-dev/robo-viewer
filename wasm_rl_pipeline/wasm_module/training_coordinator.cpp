#include "common.h"
#include "vectorized_env.cpp"

extern "C" {
    PhysicsWorld* create_physics_world(uint32_t numEnvs);
}

class CombatTrainingCoordinator {
private:
    VectorizedEnvironment* vectorizedEnv;
    PhysicsWorld* mWorld;
    int totalEnvs;
    
public:
    CombatTrainingCoordinator(int totalEnvironments = 64) 
        : totalEnvs(totalEnvironments) {
        
        // Initialize Jolt Physics World
        mWorld = create_physics_world(totalEnvs);
        
        // Initialize Vectorized Environment with the world
        vectorizedEnv = new VectorizedEnvironment(mWorld, totalEnvs);
    }
    
    ~CombatTrainingCoordinator() {
        delete vectorizedEnv;
        // The world should ideally be cleaned up too, but destroy_physics_world 
        // is in jolt_physics_bindings.cpp
    }
    
    void stepTraining(const float* actions) {
        vectorizedEnv->step(actions, totalEnvs);
    }
    
    void resetTraining() {
        vectorizedEnv->reset();
    }
    
    void getObservations(float* obs) {
        vectorizedEnv->getObservations(obs);
    }
    
    void getRewards(float* rewards) {
        vectorizedEnv->getRewards(rewards);
    }
    
    void getDones(int* dones) {
        vectorizedEnv->getDones(dones);
    }
    
    int getObservationDim() const { return 256; }
    int getActionDim() const { return 56; }
    int getTotalEnvironments() const { return totalEnvs; }
};
