#pragma once

// Robot visual state for triple buffering
struct RobotVisualState {
    float x, y, z;
    float rx, ry, rz, rw;
    float hp;
};

struct EnvVisualState {
    RobotVisualState r1, r2;
};
