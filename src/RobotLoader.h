#pragma once

#include <string>
#include <vector>

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/BodyID.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Constraints/TwoBodyConstraint.h>

struct RobotData
{
    std::vector<JPH::BodyID> bodies;
    std::vector<JPH::TwoBodyConstraint*> constraints;
};

class RobotLoader
{
public:
    RobotData LoadRobot(const std::string& filepath, JPH::PhysicsSystem* physicsSystem);
};
