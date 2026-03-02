# Battery System Tests

## Quick Start

```bash
# Run all tests
./run_tests.sh

# Or individually:
./battery_test        # Unit tests
./integration_test    # KOTH charging integration test
```

## Test Files

### 1. `battery_standalone_test.cpp` - Unit Tests (12 tests)
**Purpose:** Test BatterySystem logic in isolation (no Jolt dependency)

**Tests:**
- `battery_init_full` - Battery initializes at 100% charge
- `wireless_charging_works` - Charging works within 5m range
- `wireless_out_of_range` - No charging outside 5m range
- `regen_braking_works` - Energy recovered on deceleration
- `no_regen_accelerating` - No regen when accelerating
- `power_request_works` - Power delivery works correctly
- `power_limited_by_max` - Output limited to maxOutput (100 J/s)
- `power_setting_affects_output` - Power settings limit output
- `overheat_blocks_power` - No power when overheated
- `cooling_works` - Passive cooling reduces temperature
- `low_power_flag` - Low power flags trigger correctly
- `charge_discharge_tracking` - Statistics are tracked

**Compile:** `g++ -std=c++17 -O2 src/battery_standalone_test.cpp -o battery_test`

### 2. `integration_test.cpp` - KOTH Charging Test
**Purpose:** Verify battery charging works relative to KOTH (King of the Hill) point

**Key Feature:** Wireless charging zone moves with the KOTH target!

**Simulates:**
- KOTH point at (10, 2, 10) - NOT at arena center
- Robot1 at 2m from KOTH (within 5m charging range)
- Robot2 at 8m from KOTH (outside charging range)

**Verifies:**
- Robot1 charges when near KOTH point
- Robot2 does NOT charge when far from KOTH
- Regenerative braking works for both robots
- Charge rate follows quadratic falloff formula

**Expected Results:**
```
Robot1 (2m from KOTH):
  - Charging: YES
  - Charge rate: ~5.4 J/s (15 × (1-2/5)²)
  - After 1 second: +5.4 J from charging

Robot2 (8m from KOTH):
  - Charging: NO (out of range)
  - Charge rate: 0 J/s
```

**Compile:** `g++ -std=c++17 -O2 src/integration_test.cpp -o integration_test`

## Wireless Charging Formula

```
chargeRate = WIRELESS_CHARGE_MAX × (1 - distance/WIRELESS_CHARGE_RANGE)²

Where:
  - WIRELESS_CHARGE_MAX = 15.0 J/s (at 0m distance)
  - WIRELESS_CHARGE_RANGE = 5.0 m (max range)
  - distance = distance from robot to KOTH point
```

**Examples:**
| Distance from KOTH | Charge Rate | Notes |
|-------------------|-------------|-------|
| 0m (at KOTH) | 15.0 J/s | Maximum |
| 2m | 5.4 J/s | Typical combat range |
| 4m | 0.6 J/s | Edge of range |
| 5m+ | 0 J/s | Out of range |

## Integration Points

### CombatEnv.cpp - Reset()
```cpp
// Spawn robots on opposite sides of KOTH point with random angle
float angle = rng() * 2π;
float offsetX = cos(angle) * spawnOffset;
float offsetZ = sin(angle) * spawnOffset;

JPH::RVec3 pos1(kothPoint.x - offsetX, spawnHeight, kothPoint.z - offsetZ);
JPH::RVec3 pos2(kothPoint.x + offsetX, spawnHeight, kothPoint.z + offsetZ);
```

### CombatEnv.cpp - ManagePowerSystems()
```cpp
// Calculate distance to wireless charger (KOTH point)
JPH::RVec3 position = bodyInterface.GetPosition(robot.mainBodyId);
float chargeDistance = (position - mKothPoint).Length();

// Update battery with distance-based charging
robot.battery.Update(dt, velocity, prevVelocity, chargeDistance);
```

## Battery Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `BATTERY_DEFAULT_CAPACITY` | 1000.0f | Max energy (Joules) |
| `WIRELESS_CHARGE_RANGE` | 5.0f | Max charging distance (meters) |
| `WIRELESS_CHARGE_MAX` | 15.0f | Max charge rate at 0m (J/s) |
| `BATTERY_REGEN_EFFICIENCY` | 0.3f | 30% energy recovery |
| `BATTERY_DEFAULT_OUTPUT` | 100.0f | Max discharge rate (J/s) |
| `BATTERY_OVERHEAT_TEMP` | 80.0f | Overheat cutoff (°C) |

## Ability Costs

| Ability | Action Index | Cost | Type |
|---------|--------------|------|------|
| EMP | 52 | 150.0f | Attack |
| Shield | 53 | 25.0f/s | Shield |
| Slow-mo | 54 | 50.0f/s | General |
| Power Setting | 55 | N/A | Multiplier (0.1-1.0) |

## Debugging

### If charging seems weak:

1. **Check distance to KOTH:**
   - Charging only works within 5m of KOTH point
   - Charge rate drops quadratically with distance
   - At 4m: only 0.6 J/s (barely noticeable)

2. **Verify KOTH position:**
   - KOTH point randomizes each reset
   - Check `mKothPoint` in debugger
   - Robots spawn relative to KOTH, not arena center

3. **Check battery state:**
   - `isCharging` flag should be true when in range
   - `currentCharge` shows instantaneous charge rate (J/s)
   - `totalCharged` accumulates over time

### Expected charge values in UI:

| Scenario | currentCharge | Notes |
|----------|---------------|-------|
| At KOTH (0m) | ~15.0 J/s | Maximum |
| Near KOTH (2m) | ~5.4 J/s | Common |
| Edge (4m) | ~0.6 J/s | Very weak |
| Out of range | 0.0 J/s | No charging |

## Regenerative Braking

**Formula:**
```
recovered = 0.5 × mass × (prevSpeed² - currSpeed²) × regenEfficiency

Where:
  - mass = 100 kg (robot mass)
  - regenEfficiency = 0.3 (30%)
```

**Example:**
- Braking from 5 m/s to 2 m/s
- Kinetic energy change: 0.5 × 100 × (25 - 4) = 1050 J
- Recovered: 1050 × 0.3 = **315 J**

This is why you see large `totalRegenerated` values during combat!
