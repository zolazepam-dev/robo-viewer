// Engine Battery Drain & CoG Test
#include <iostream>
#include <cmath>
#include <iomanip>

namespace JPH {
    class Vec3 {
    public:
        float x, y, z;
        Vec3() : x(0), y(0), z(0) {}
        Vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
        float Length() const { return std::sqrt(x*x + y*y + z*z); }
        float GetX() const { return x; }
        float GetY() const { return y; }
        float GetZ() const { return z; }
        Vec3 operator*(float s) const { return Vec3(x*s, y*s, z*s); }
        Vec3 operator+(const Vec3& o) const { return Vec3(x+o.x, y+o.y, z+o.z); }
        Vec3& operator+=(const Vec3& o) { x+=o.x; y+=o.y; z+=o.z; return *this; }
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
    float movementPowerSetting = 1.0f;
    float shieldPowerSetting = 1.0f;
    
    void Reset(const BatteryConfig& cfg) {
        currentEnergy = cfg.capacity;
        temperature = cfg.ambientTemp;
        isOverheated = false;
        isCharging = false;
        isDischarging = false;
        currentDraw = 0.0f;
        currentCharge = 0.0f;
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
        switch (type) {
            case PowerType::Attack: setting = mState.attackPowerSetting; break;
            case PowerType::Movement: setting = mState.movementPowerSetting; break;
            case PowerType::Shield: setting = mState.shieldPowerSetting; break;
            case PowerType::General: setting = 1.0f; break;
        }
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
    
    const BatteryState& GetState() const { return mState; }
private:
    BatteryConfig mConfig;
    BatteryState mState;
};

// Simulate engine thrust power consumption
void SimulateEngineThrust(BatterySystem& battery, float totalThrust, float dt) {
    const float THRUST_EFFICIENCY = 0.1f;  // J/s per Newton
    float powerNeeded = totalThrust * THRUST_EFFICIENCY;
    battery.RequestPower(powerNeeded, PowerType::Movement, dt);
}

// Simulate CoG calculation
JPH::Vec3 CalculateCoG(const JPH::Vec3* enginePositions, float engineMass = 15.0f, float shellMass = 50.0f) {
    JPH::Vec3 totalMoment(0, 0, 0);
    float totalMass = shellMass;
    
    for (int i = 0; i < 3; i++) {
        totalMoment += enginePositions[i] * engineMass;
        totalMass += engineMass;
    }
    
    return JPH::Vec3(totalMoment.x / totalMass, totalMoment.y / totalMass, totalMoment.z / totalMass);
}

int main() {
    std::cout << std::fixed << std::setprecision(2);
    
    std::cout << "\n╔════════════════════════════════════════════╗\n";
    std::cout << "║  ENGINE BATTERY DRAIN & CoG TRACKING TEST ║\n";
    std::cout << "╚════════════════════════════════════════════╝\n\n";
    
    BatterySystem battery;
    battery.Init();
    
    const float dt = 1.0f / 60.0f;
    
    // Test 1: Engine thrust drains battery
    std::cout << "TEST 1: Engine Thrust Battery Drain\n";
    std::cout << "─────────────────────────────────────\n";
    
    float initialEnergy = battery.GetState().currentEnergy;
    std::cout << "Initial Energy: " << initialEnergy << " J\n\n";
    
    // Simulate 1 second of thrust at various levels
    struct ThrustTest { float thrust; const char* desc; };
    ThrustTest tests[] = {
        {0.0f, "Idle (no thrust)"},
        {50.0f, "Low thrust (50N)"},
        {150.0f, "Medium thrust (150N)"},
        {300.0f, "High thrust (300N)"},
        {500.0f, "Max thrust (500N)"}
    };
    
    for (auto& test : tests) {
        battery.Init();  // Reset
        for (int i = 0; i < 60; i++) {
            SimulateEngineThrust(battery, test.thrust, dt);
        }
        float consumed = initialEnergy - battery.GetState().currentEnergy;
        float expectedPower = test.thrust * 0.1f;  // 0.1 J/s per N
        float expectedEnergy = expectedPower * 1.0f;  // 1 second
        
        std::cout << test.desc << ":\n";
        std::cout << "  Expected drain: " << expectedEnergy << " J\n";
        std::cout << "  Actual drain:   " << consumed << " J\n";
        std::cout << "  Final Energy:   " << battery.GetState().currentEnergy << " J\n";
        std::cout << "  Discharged:     " << battery.GetState().totalDischarged << " J\n\n";
    }
    
    // Test 2: CoG calculation
    std::cout << "\nTEST 2: Center of Gravity Tracking\n";
    std::cout << "─────────────────────────────────────\n\n";
    
    // Symmetric engine placement (CoG at center)
    JPH::Vec3 symmetricEngines[3] = {
        JPH::Vec3(-0.1f, 0.0f, 0.0f),
        JPH::Vec3(0.1f, 0.0f, 0.0f),
        JPH::Vec3(0.0f, 0.0f, 0.0f)
    };
    JPH::Vec3 symCoG = CalculateCoG(symmetricEngines);
    float symDist = symCoG.Length();
    
    std::cout << "Symmetric placement:\n";
    std::cout << "  Engine positions: (-0.1, 0, 0), (0.1, 0, 0), (0, 0, 0)\n";
    std::cout << "  CoG: (" << symCoG.GetX() << ", " << symCoG.GetY() << ", " << symCoG.GetZ() << ") m\n";
    std::cout << "  Displacement: " << symDist << " m (should be ~0)\n\n";
    
    // Asymmetric engine placement (CoG offset)
    JPH::Vec3 asymmetricEngines[3] = {
        JPH::Vec3(0.3f, 0.0f, 0.0f),
        JPH::Vec3(0.3f, 0.0f, 0.0f),
        JPH::Vec3(0.3f, 0.0f, 0.0f)
    };
    JPH::Vec3 asymCoG = CalculateCoG(asymmetricEngines);
    float asymDist = asymCoG.Length();
    
    std::cout << "Asymmetric placement:\n";
    std::cout << "  Engine positions: (0.3, 0, 0) x3\n";
    std::cout << "  CoG: (" << asymCoG.GetX() << ", " << asymCoG.GetY() << ", " << asymCoG.GetZ() << ") m\n";
    std::cout << "  Displacement: " << asymDist << " m (should be ~0.3 * 45/95)\n\n";
    
    // Test 3: Combined thrust + CoG
    std::cout << "\nTEST 3: Combined Scenario\n";
    std::cout << "─────────────────────────────────────\n";
    
    battery.Init();
    float totalThrust = 200.0f;
    int steps = 60;
    
    for (int i = 0; i < steps; i++) {
        SimulateEngineThrust(battery, totalThrust, dt);
    }
    
    std::cout << "After " << steps << " steps (1 second) at " << totalThrust << "N thrust:\n";
    std::cout << "  Energy consumed: " << (1000.0f - battery.GetState().currentEnergy) << " J\n";
    std::cout << "  Total discharged: " << battery.GetState().totalDischarged << " J\n";
    std::cout << "  Movement power setting: " << battery.GetState().movementPowerSetting << "\n\n";
    
    std::cout << "╔════════════════════════════════════════════╗\n";
    std::cout << "║              TEST COMPLETE                ║\n";
    std::cout << "╚════════════════════════════════════════════╝\n\n";
    
    return 0;
}
