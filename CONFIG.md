# Runtime Configuration (game_config.json)

Location: `config/game_config.json`

## Fields and defaults
```jsonc
{
  "damage": {
    "multiplier": 10.0,
    "spike_threshold": 0.55,
    "engine_threshold": 1.0,
    "spike_scale": 0.001,
    "engine_scale": 0.002,
    "spike_floor": 0.0,
    "engine_floor": 0.0
  },
  "hp": { "initial": 100.0 },
  "koth": {
    "radius": 0.8,
    "random_xz_min": -15.0,
    "random_xz_max": 15.0,
    "random_y_min": 2.0,
    "random_y_max": 12.0
  },
  "env": {
    "max_steps": 7200,
    "spawn_offset": 2.5,
    "spawn_height": 5.0,
    "initial_push_speed": 8.0
  },
  "reward": {
    "proximity_range": 15.0,
    "proximity_scale": 0.1,
    "proximity_far_penalty": 0.05,
    "approach_scale": 0.02,
    "wall_start": 14.0,
    "wall_penalty_scale": 0.1,
    "energy_scale": 0.01,
    "koth_win": 0.1,
    "altitude_scale": 0.05
  },
  "debug": {
    "enable_damage_logs": false,
    "hp_threshold_logs": [75, 50, 25]
  }
}
```

## How it is applied
- Loaded at startup in `main_train.cpp` and `main_train_headless.cpp`. Defaults apply if the file is missing or malformed.
- Values are cached in a `Config` POD (no per-step allocations). Hot paths read plain floats.
- `CombatEnv` uses config for HP, KOTH ranges, spawns, episode length, and damage thresholds/scales.

## How to change
1) Edit `config/game_config.json` (JSON format, see schema above).
2) Run viewer or train normally: `bazel run //:viewer` or `bazel run //:train --config=opt`.
3) No rebuild needed if only the JSON changes; binaries read the file at startup.

## Notes and tips
- Keep thresholds/scales consistent with your robot sizes; large values may cause constant damage hits.
- If HP never drops, temporarily raise multipliers or thresholds to test the pipeline.
- Debug logs (`enable_damage_logs`) should stay false in production to preserve FPS.
