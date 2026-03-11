/**
 * @file OctopodLoader.h
 * @brief Loader for octopod robot in native Jolt format
 * 
 * Loads the octopod.json file which uses native Jolt Physics API mapping.
 * The octopod has 25 bodies and 24 hinge constraints.
 */

#pragma once

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <string>
#include <vector>
#include <map>

namespace OctopodLoader {

/**
 * @brief Load octopod robot from JSON file
 * @param configPath Path to octopod.json
 * @param physicsSystem Pointer to physics system
 * @param spawnPosition Spawn position for the robot
 * @param envIndex Environment index for collision layers
 * @return Loaded octopod structure with bodies and constraints
 */
struct LoadedOctopod {
    std::vector<JPH::BodyID> bodies;
    std::vector<JPH::HingeConstraint*> constraints;
    std::map<std::string, JPH::BodyID> bodyMap;
    JPH::BodyID centralBody;
};

LoadedOctopod LoadOctopod(
    const std::string& configPath,
    JPH::PhysicsSystem* physicsSystem,
    const JPH::RVec3& spawnPosition,
    uint32_t envIndex,
    JPH::GroupFilter* groupFilter = nullptr);

/**
 * @brief Apply motor actions to hinge constraints
 * @param constraints Vector of hinge constraints
 * @param actions Action values (normalized [-1, 1])
 * @param actionScale Scale factor for actions
 */
void ApplyMotorActions(
    std::vector<JPH::HingeConstraint*>& constraints,
    const std::vector<float>& actions,
    float actionScale = 1.0f);

/**
 * @brief Get observations from octopod state
 * @param octopod Loaded octopod structure
 * @param physicsSystem Pointer to physics system
 * @param observations Output observation array
 * @param obsDim Observation dimension
 */
void GetObservations(
    const LoadedOctopod& octopod,
    JPH::PhysicsSystem* physicsSystem,
    float* observations,
    int obsDim);

} // namespace OctopodLoader
