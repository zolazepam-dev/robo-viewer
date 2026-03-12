# Robot Templates - Arm & Snake Robots

## Overview
Programmatic robot building framework with pre-built templates for **arm manipulators** and **snake robots**. No URDF imports, no manual JSON building - pure C++ constructors.

## Files Created

### Core Framework
- `src/RobotTemplates.h` - Fluent builder API + template declarations
- `src/RobotTemplates.cpp` - Implementation of all robot templates
- `src/viewer_robots.cpp` - Standalone test viewer (separate from main train)

### Robot Configs (JSON reference)
- `robots/arm_3dof.json` - 3-DOF planar arm
- `robots/arm_5dof.json` - 5-DOF articulated arm
- `robots/snake_4seg.json` - 4-segment snake
- `robots/snake_8seg.json` - 8-segment snake

## Robot Templates

### Arm Robots

| Template | DOF | Bodies | Joints | Torque Range | Use Case |
|----------|-----|--------|--------|--------------|----------|
| `CreateArm3DOF()` | 3 | 4 | 3 hinge | 50-150 Nm | Basic reaching, planar manipulation |
| `CreateArm5DOF()` | 5 | 6 | 5 hinge | 50-200 Nm | 3D manipulation, pick & place |
| `CreateArm7DOF()` | 7 | 7 | 7 hinge | 40-250 Nm | Redundant manipulation, human-like |
| `CreateArmWithGripper()` | 5+2 | 9 | 5 hinge + 2 slider | 50-200 Nm | Object grasping |

### Snake Robots

| Template | Segments | Bodies | Joints | Torque | Use Case |
|----------|----------|--------|--------|--------|----------|
| `CreateSnake4Segment()` | 4 | 5 | 4 hinge | 20-30 Nm | Basic undulation |
| `CreateSnake8Segment()` | 8 | 9 | 8 hinge | 15-25 Nm | Smooth locomotion |
| `CreateSnake12Segment()` | 12 | 13 | 12 hinge | 10-20 Nm | Complex terrain |
| `CreateSnakeWithHead()` | 8 | 10 | 8 hinge | 15-30 Nm | Sensory head |

## Usage

### 1. Test Viewer (Recommended First Step)

```bash
bazel run //:viewer_robots
```

**Controls:**
- **1-3**: View Arm (3DOF/5DOF/7DOF)
- **4-6**: View Snake (4/8/12 segment)
- **7**: Arm with Gripper
- **8**: Snake with Head
- **R**: Reset robot
- **ESC**: Exit

### 2. Programmatic Usage

```cpp
#include "src/RobotTemplates.h"

// Create a 5-DOF arm at position (0, 5, 0)
CombatRobotData arm = RobotTemplates::CreateArm5DOF(
    &physicsSystem,
    JPH::RVec3(0.0f, 5.0f, 0.0f),
    0  // environment index
);

// Create an 8-segment snake
CombatRobotData snake = RobotTemplates::CreateSnake8Segment(
    &physicsSystem,
    JPH::RVec3(0.0f, 2.0f, 0.0f),
    0
);
```

### 3. Custom Robot Builder

```cpp
#include "src/RobotTemplates.h"

RobotBuilder builder;
builder.SetName("custom_robot")
       .SetSpawnPos(JPH::RVec3(0, 5, 0))
       .SetEnvIndex(0)
       .SetRobotIndex(0);

// Add bodies
builder.AddBoxBody("base", JPH::Vec3(0.3f, 0.1f, 0.3f), 5.0f);
builder.AddCapsuleBody("arm", 0.08f, 0.3f, 2.0f, JPH::Vec3(0, 0.4f, 0));

// Add joints
builder.AddHingeJoint("base", "arm", JPH::Vec3::sAxisZ(), 100.0f, 5.0f);

// Build
CombatRobotData robot = builder.Build(&physicsSystem);
```

## Robot Specifications

### Arm 5-DOF (Example)

```
Segment        | Mass (kg) | Size/Radius    | Length (m)
---------------|-----------|----------------|------------
Base           | 8.0       | 0.8×0.3×0.8    | 0.3
Shoulder       | 3.0       | 0.15 radius    | 0.2
Lower Arm      | 2.5       | 0.08 radius    | 0.7
Upper Arm      | 2.0       | 0.07 radius    | 0.6
Forearm        | 1.5       | 0.06 radius    | 0.4
Hand           | 0.8       | 0.24×0.12×0.36 | 0.12

Total Height: ~2.4m
Total Mass: ~17.8 kg
```

### Snake 8-Segment (Example)

```
Segment | Mass (kg) | Radius (m) | Length (m) | Joint Torque (Nm)
--------|-----------|------------|------------|------------------
Head    | 1.56      | 0.08       | 0.49       | -
Neck    | 1.2       | 0.08       | 0.7        | 25
Seg 0-5 | 1.2       | 0.08       | 0.7        | 25
Tail    | 0.48      | 0.056      | 0.28       | 15

Total Length: ~5.6m
Total Mass: ~10.3 kg
```

## Control Interfaces

### Arm Control (5-DOF)
```cpp
// Action space: [shoulder_pan, shoulder_tilt, elbow, wrist_tilt, wrist_rotate]
// Range: [-1, 1] normalized, scaled by joint speed limits
actions[0] = shoulder pan velocity (±4 rad/s)
actions[1] = shoulder tilt velocity (±6 rad/s)
actions[2] = elbow velocity (±8 rad/s)
actions[3] = wrist tilt velocity (±10 rad/s)
actions[4] = wrist rotate velocity (±12 rad/s)
```

### Snake Control (8-Segment)
```cpp
// Action space: [joint0, joint1, ..., joint7]
// Gait: Sine wave undulation
// Can implement central pattern generator (CPG) for natural motion
actions[i] = joint i angular velocity (±6 rad/s)

// Example sine wave gait:
for (int i = 0; i < 8; i++) {
    actions[i] = A * sin(ω*t + phase_offset*i);
}
```

## Physics Properties

| Property | Arms | Snakes |
|----------|------|--------|
| Friction | 0.5 | 0.3 |
| Restitution | 0.1 | 0.05 |
| Linear Damping | 0.1 | 0.15 |
| Angular Damping | 0.1 | 0.15 |

## Integration with Training

### Option 1: Direct Usage
```cpp
// In your env initialization
arm = RobotTemplates::CreateArm5DOF(&physics, spawnPos, envIndex);
```

### Option 2: JSON Config Loading
Use the provided JSON files with your existing `RobotLoader`

### Option 3: Hybrid
Load base config from JSON, then modify with builder:
```cpp
RobotBuilder builder;
// ... configure from JSON ...
// Add custom modifications
builder.AddBoxBody("sensor_mount", ...);
```

## Next Steps

1. **Test the viewer**: `bazel run //:viewer_robots`
2. **Pick a robot**: Choose based on your task complexity
3. **Define rewards**: Position tracking, object manipulation, locomotion
4. **Train**: Integrate into your RL environment

## Task Ideas

### Arms
- **Reaching**: Move end-effector to target position
- **Tracking**: Follow moving target
- **Writing**: Draw shapes with end-effector
- **Pushing**: Move object to goal
- **Stacking**: Stack blocks

### Snakes
- **Locomotion**: Move forward as fast as possible
- **Following**: Follow a path/trajectory
- **Obstacle avoidance**: Navigate around obstacles
- **Terrain**: Climb over rough terrain
- **Target reaching**: Move head to target

## Build Commands

```bash
# Build viewer
bazel build //:viewer_robots

# Run viewer
bazel run //:viewer_robots

# Build (automatically includes templates in core library)
bazel build //:train
```
