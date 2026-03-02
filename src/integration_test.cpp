// Integration Test - KOTH-based charging
#include <iostream>
#include <cmath>
#include <cstdint>

namespace JPH {
    class Vec3 {
    public:
        float x, y, z;
        Vec3() : x(0), y(0), z(0) {}
        Vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
        float Length() const { return std::sqrt(x*x + y*y + z*z); }
    };
    class RVec3 {
    public:
        double x, y, z;
        RVec3() : x(0), y(0), z(0) {}
        RVec3(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}
        double Length() const { return std::sqrt(x*x + y*y + z*z); }
        RVec3 operator-(const RVec3& o) const { return RVec3(x-o.x, y-o.y, z-o.z); }
    };
    class BodyID {
    public:
        uint32_t index = 0xFFFFFFFF;
        bool IsInvalid() const { return index == 0xFFFFFFFF; }
    };
}

constexpr float BATTERY_DEFAULT_CAPACITY = 1000.0f;
constexpr float BATTERY_AMBIENT_TEMP = 20.0f;
constexpr float BATTERY_OVERHEAT_TEMP = 80.0f;
constexpr float WIRELESS_CHARGE_RANGE = 5.0f;
constexpr float WIRELESS_CHARGE_MAX = 15.0f;
constexpr float BATTERY_REGEN_EFFICIENCY = 0.3f;
constexpr float BATTERY_DEFAULT_HEAT_CAPACITY = 100.0f;
constexpr float BATTERY_DEFAULT_COOLING = 5.0f;
constexpr float EMP_COST = 150.0f;

enum class PowerType { Attack, Movement, Shield, General };

struct BatteryConfig {
    float capacity = BATTERY_DEFAULT_CAPACITY;
    float wirelessRange = WIRELESS_CHARGE_RANGE;
    float wirelessMaxRate = WIRELESS_CHARGE_MAX;
    float regenEfficiency = BATTERY_REGEN_EFFICIENCY;
    float heatCapacity = BATTERY_DEFAULT_HEAT_CAPACITY;
    float coolingRate = BATTERY_DEFAULT_COOLING;
    float overheatTemp = BATTERY_OVERHEAT_TEMP;
    float ambientTemp = BATTERY_AMBIENT_TEMP;
    float maxOutput = 100.0f;
};

struct BatteryState {
    float currentEnergy = BATTERY_DEFAULT_CAPACITY;
    float totalCharged = 0.0f;
    float totalDischarged = 0.0f;
    float totalRegenerated = 0.0f;
    float temperature = BATTERY_AMBIENT_TEMP;
    bool isOverheated = false;
    bool isCharging = false;
    bool isDischarging = false;
    float currentDraw = 0.0f;
    float currentCharge = 0.0f;
    float attackPowerSetting = 1.0f;
    
    void Reset(const BatteryConfig& cfg) {
        currentEnergy = cfg.capacity * 0.5f;
        temperature = cfg.ambientTemp;
        isOverheated = false;
        isCharging = false;
    }
};

class BatterySystem {
public:
    BatterySystem() { Init(); }
    void Init(const BatteryConfig& cfg = BatteryConfig()) { mConfig = cfg; mState.Reset(cfg); }
    
    void Update(float dt, const JPH::Vec3& velocity, const JPH::Vec3& prevVelocity, float wirelessChargeDistance = 1000.0f) {
        mState.isCharging = false;
        mState.isDischarging = false;
        mState.currentDraw = 0.0f;
        mState.currentCharge = 0.0f;
        
        if (wirelessChargeDistance < mConfig.wirelessRange && !mState.isOverheated) {
            float chargeFactor = 1.0f - (wirelessChargeDistance / mConfig.wirelessRange);
            float chargeRate = mConfig.wirelessMaxRate * chargeFactor * chargeFactor;
            mState.currentEnergy = fminf(mState.currentEnergy + chargeRate * dt, mConfig.capacity);
            mState.currentCharge = chargeRate;
            mState.isCharging = true;
            mState.totalCharged += chargeRate * dt;
        }
        
        float currentSpeed = velocity.Length();
        float prevSpeed = prevVelocity.Length();
        if (currentSpeed < prevSpeed && currentSpeed > 0.1f && !mState.isOverheated) {
            float kineticEnergy = 0.5f * 100.0f * (prevSpeed * prevSpeed - currentSpeed * currentSpeed);
            float recovered = kineticEnergy * mConfig.regenEfficiency;
            mState.currentEnergy = fminf(mState.currentEnergy + recovered, mConfig.capacity);
            mState.totalRegenerated += recovered;
        }
        
        float tempDiff = mState.temperature - mConfig.ambientTemp;
        float heatDissipated = mConfig.coolingRate * tempDiff * dt;
        mState.temperature = fmaxf(mConfig.ambientTemp, mState.temperature - heatDissipated / mConfig.heatCapacity);
        
        if (mState.temperature >= mConfig.overheatTemp) mState.isOverheated = true;
        if (mState.isOverheated && mState.temperature < (mConfig.overheatTemp - 10.0f)) mState.isOverheated = false;
    }
    
    float RequestPower(float requested, PowerType type, float dt) {
        if (mState.isOverheated) return 0.0f;
        float setting = 1.0f;
        float maxAvailable = fminf(mConfig.maxOutput * setting, mState.currentEnergy / dt);
        float actual = fminf(requested, maxAvailable);
        if (actual > 0.0f) {
            mState.currentEnergy -= actual * dt;
            mState.currentDraw = actual;
            mState.isDischarging = true;
            mState.totalDischarged += actual * dt;
        }
        return actual;
    }
    
    void SetAttackPowerSetting(float s) { mState.attackPowerSetting = fmaxf(0.1f, fminf(1.0f, s)); }
    const BatteryState& GetState() const { return mState; }
private:
    BatteryConfig mConfig;
    BatteryState mState;
};

struct CombatRobotData {
    JPH::BodyID mainBodyId;
    BatterySystem battery;
    float hp = 100.0f;
    CombatRobotData() { mainBodyId.index = 1; }
};

void SimulateManagePowerSystems(CombatRobotData& robot, const float* actions, float dt, const JPH::RVec3& kothPoint, const JPH::RVec3& robotPos) {
    if (robot.mainBodyId.IsInvalid()) return;
    
    JPH::Vec3 velocity(3, 0, 0);
    JPH::Vec3 prevVelocity(5, 0, 0);
    
    // Calculate distance to KOTH point (wireless charger)
    float chargeDistance = static_cast<float>((robotPos - kothPoint).Length());
    
    robot.battery.Update(dt, velocity, prevVelocity, chargeDistance);
    
    if (actions[52] > 0.5f) {
        robot.battery.RequestPower(EMP_COST, PowerType::Attack, dt);
    }
    
    float powerSetting = fmaxf(0.1f, fminf(1.0f, actions[55]));
    robot.battery.SetAttackPowerSetting(powerSetting);
}

int main() {
    std::cout << "\n\033[1;36m╔════════════════════════════════════════════╗\033[0m\n";
    std::cout << "\033[1;36m║  KOTH-BASED CHARGING INTEGRATION TEST     ║\033[0m\n";
    std::cout << "\033[1;36m╚════════════════════════════════════════════╝\033[0m\n\n";
    
    CombatRobotData robot1, robot2;
    float actions1[56] = {0};
    float actions2[56] = {0};
    
    // KOTH point at (10, 2, 10) - not at arena center!
    JPH::RVec3 kothPoint(10.0, 2.0, 10.0);
    
    // Robot1 spawns 2m from KOTH (should charge)
    JPH::RVec3 robot1Pos(kothPoint.x + 2.0, 2.0, kothPoint.z);
    
    // Robot2 spawns 8m from KOTH (out of range, should NOT charge)
    JPH::RVec3 robot2Pos(kothPoint.x + 8.0, 2.0, kothPoint.z);
    
    actions1[55] = 0.7f;  // Power setting
    
    std::cout << "KOTH Point (charging zone): (10, 2, 10)\n";
    std::cout << "Robot1 position: (12, 2, 10) - 2m from KOTH\n";
    std::cout << "Robot2 position: (18, 2, 10) - 8m from KOTH (out of range)\n\n";
    
    std::cout << "Initial State (50% battery):\n";
    std::cout << "  Robot1 Energy: " << robot1.battery.GetState().currentEnergy << " J\n";
    std::cout << "  Robot2 Energy: " << robot2.battery.GetState().currentEnergy << " J\n\n";
    
    const float dt = 1.0f / 60.0f;
    
    // Simulate 60 game steps (1 second)
    for (int i = 0; i < 60; i++) {
        SimulateManagePowerSystems(robot1, actions1, dt, kothPoint, robot1Pos);
        SimulateManagePowerSystems(robot2, actions2, dt, kothPoint, robot2Pos);
    }
    
    const auto& s1 = robot1.battery.GetState();
    const auto& s2 = robot2.battery.GetState();
    
    std::cout << "After 60 Steps (1 second):\n";
    std::cout << "  Robot1 Energy: " << s1.currentEnergy << " J (was 500 J)\n";
    std::cout << "  Robot1 Charging: " << (s1.isCharging ? "YES ✓" : "NO ✗") << "\n";
    std::cout << "  Robot1 Total Charged: " << s1.totalCharged << " J\n";
    std::cout << "  Robot1 Total Regen: " << s1.totalRegenerated << " J\n\n";
    
    std::cout << "  Robot2 Energy: " << s2.currentEnergy << " J (was 500 J)\n";
    std::cout << "  Robot2 Charging: " << (s2.isCharging ? "YES ✗" : "NO ✓") << "\n";
    std::cout << "  Robot2 Total Charged: " << s2.totalCharged << " J\n\n";
    
    // Verify
    bool pass = true;
    
    // Robot1 should be charging (2m from KOTH, within 5m range)
    if (!s1.isCharging) {
        std::cout << "\033[31m✗ FAIL: Robot1 should be charging (2m from KOTH)\033[0m\n";
        pass = false;
    }
    if (s1.totalCharged < 4.0f) {
        std::cout << "\033[31m✗ FAIL: Robot1 should have charged ~5.4 J\033[0m\n";
        pass = false;
    }
    
    // Robot2 should NOT be charging (8m from KOTH, outside 5m range)
    if (s2.isCharging) {
        std::cout << "\033[31m✗ FAIL: Robot2 should NOT charge (8m from KOTH)\033[0m\n";
        pass = false;
    }
    if (s2.totalCharged > 0.1f) {
        std::cout << "\033[31m✗ FAIL: Robot2 should not have charged\033[0m\n";
        pass = false;
    }
    
    // Both should have regen from braking
    if (s1.totalRegenerated <= 0 || s2.totalRegenerated <= 0) {
        std::cout << "\033[31m✗ FAIL: Both robots should have regen energy\033[0m\n";
        pass = false;
    }
    
    if (pass) {
        std::cout << "\033[32m✓ KOTH CHARGING TEST PASSED\033[0m\n";
        std::cout << "  ✓ Wireless charging works relative to KOTH point\n";
        std::cout << "  ✓ Out-of-range robots don't charge\n";
        std::cout << "  ✓ Regenerative braking works\n";
    } else {
        std::cout << "\033[31m✗ KOTH CHARGING TEST FAILED\033[0m\n";
    }
    
    std::cout << "\n";
    return pass ? 0 : 1;
}
