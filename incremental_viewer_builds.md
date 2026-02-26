# Incremental Viewer Builds

## Sphere Demo
To build and run the single sphere physics demo:
```bash
bazel run //sphere_demo:sphere_demo
```

## Room Demo (Pro Viewer)
Large room with transparent walls, 2 robots, and professional controls:
- **Free Camera**: Hold Right-Click + WASD (QE for vertical)
- **Time Controls**: Real-time simulation speed (Time Scale)
- **Physics Tuner**: Live adjustment of solver steps and stability constants.
```bash
bazel run //room_demo:room_demo
```

## Robot Demo
To build and run the combat robot with 13 satellites:
```bash
bazel run //robot_demo:robot_demo
```

## Random Robot Demo
To build and run the combat robot with random actions:
```bash
bazel run //random_robot_demo:random_robot_demo
```

## Random Policy Room Demo
Subversion of the Pro Room Demo where both robots execute random actions every frame.
```bash
bazel run //random_room_demo:random_room_demo
```

## Full Training Viewer
The complete training setup with both agents, free camera, and live tuning of physics and power.
```bash
bazel run //:viewer
```
