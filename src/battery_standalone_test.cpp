// Standalone Battery Test - No game launch required
// Compile: g++ -std=c++17 -I. -Ithird_party/jolt src/battery_standalone_test.cpp -o battery_test
// Run: ./battery_test

#include <iostream>
#include <cmath>
#include <cstring>
#include <iomanip>

// Minimal Jolt Vec3 stub for testing
namespace JPH {
    class Vec3 {
    public:
        float x, y, z;
        Vec3() : x(0), y(0), z(0) {}
        Vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
        float Length() const { return std::sqrt(x*x + y*y + z*z); }
        Vec3 operator-(const Vec3& o) const { return Vec3(x-o.x, y-o.y, z-o.z); }
    };
    class RVec3 {
    public:
        double x, y, z;
        RVec3() : x(0), y(0), z(0) {}
        RVec3(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}
        double Length() const { return std::sqrt(x*x + y*y + z*z); }
    };
}

// Inline the entire BatterySystem for standalone testing
constexpr float BATTERY_DEFAULT_CAPACITY = 1000.0f;
constexpr float BATTERY_DEFAULT_CHARGE_RATE = 20.0f;
constexpr float BATTERY_DEFAULT_OUTPUT = 100.0f;
constexpr float BATTERY_DEFAULT_HEAT_CAPACITY = 100.0f;
constexpr float BATTERY_DEFAULT_COOLING = 5.0f;
constexpr float BATTERY_AMBIENT_TEMP = 20.0f;
constexpr float BATTERY_OVERHEAT_TEMP = 80.0f;
constexpr float BATTERY_REGEN_EFFICIENCY = 0.3f;
constexpr float WIRELESS_CHARGE_RANGE = 5.0f;
constexpr float WIRELESS_CHARGE_MAX = 15.0f;

enum class PowerType { Attack, Movement, Shield, General };

struct BatteryConfig {
    float capacity = BATTERY_DEFAULT_CAPACITY;
    float chargeRate = BATTERY_DEFAULT_CHARGE_RATE;
    float maxOutput = BATTERY_DEFAULT_OUTPUT;
    float heatCapacity = BATTERY_DEFAULT_HEAT_CAPACITY;
    float coolingRate = BATTERY_DEFAULT_COOLING;
    float overheatTemp = BATTERY_OVERHEAT_TEMP;
    float ambientTemp = BATTERY_AMBIENT_TEMP;
    float regenEfficiency = BATTERY_REGEN_EFFICIENCY;
    float wirelessRange = WIRELESS_CHARGE_RANGE;
    float wirelessMaxRate = WIRELESS_CHARGE_MAX;
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
        totalCharged = 0.0f;
        totalDischarged = 0.0f;
        totalRegenerated = 0.0f;
    }
    
    float GetChargePercent() const { return (currentEnergy / BATTERY_DEFAULT_CAPACITY) * 100.0f; }
    bool IsLowPower() const { return currentEnergy < (BATTERY_DEFAULT_CAPACITY * 0.2f); }
};

class BatterySystem {
public:
    void Init(const BatteryConfig& cfg = BatteryConfig()) {
        mConfig = cfg;
        mState.Reset(cfg);
    }
    
    void Update(float dt, const JPH::Vec3& velocity, const JPH::Vec3& prevVelocity, 
                float wirelessChargeDistance = 1000.0f) {
        mState.isCharging = false;
        mState.isDischarging = false;
        mState.currentDraw = 0.0f;
        mState.currentCharge = 0.0f;
        
        // 1. Wireless charging
        if (wirelessChargeDistance < mConfig.wirelessRange && !mState.isOverheated) {
            float chargeFactor = 1.0f - (wirelessChargeDistance / mConfig.wirelessRange);
            float chargeRate = mConfig.wirelessMaxRate * chargeFactor * chargeFactor;
            float heatGen = chargeRate * 0.1f;
            
            mState.currentEnergy = fminf(mState.currentEnergy + chargeRate * dt, mConfig.capacity);
            mState.currentCharge = chargeRate;
            mState.isCharging = true;
            mState.totalCharged += chargeRate * dt;
            mState.totalRegenerated += heatGen * dt;
            mState.temperature += heatGen * dt / mConfig.heatCapacity;
        }
        
        // 2. Regenerative braking
        float currentSpeed = velocity.Length();
        float prevSpeed = prevVelocity.Length();
        if (currentSpeed < prevSpeed && currentSpeed > 0.1f && !mState.isOverheated) {
            float kineticEnergy = 0.5f * 100.0f * (prevSpeed * prevSpeed - currentSpeed * currentSpeed);
            float recovered = kineticEnergy * mConfig.regenEfficiency;
            
            mState.currentEnergy = fminf(mState.currentEnergy + recovered, mConfig.capacity);
            mState.totalRegenerated += recovered;
        }
        
        // 3. Passive cooling
        float tempDiff = mState.temperature - mConfig.ambientTemp;
        float heatDissipated = mConfig.coolingRate * tempDiff * dt;
        mState.temperature = fmaxf(mConfig.ambientTemp, mState.temperature - heatDissipated / mConfig.heatCapacity);
        
        // 4. Overheat check
        if (mState.temperature >= mConfig.overheatTemp) {
            mState.isOverheated = true;
        }
        
        // 5. Auto-recover from overheat
        if (mState.isOverheated && mState.temperature < (mConfig.overheatTemp - 10.0f)) {
            mState.isOverheated = false;
        }
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
            
            float heatGen = actual * 0.15f;
            mState.temperature += heatGen * dt / mConfig.heatCapacity;
        }
        
        return actual;
    }
    
    void SetAttackPowerSetting(float s) { mState.attackPowerSetting = fmaxf(0.1f, fminf(1.0f, s)); }
    void SetMovementPowerSetting(float s) { mState.movementPowerSetting = fmaxf(0.1f, fminf(1.0f, s)); }
    void SetShieldPowerSetting(float s) { mState.shieldPowerSetting = fmaxf(0.1f, fminf(1.0f, s)); }
    
    const BatteryState& GetState() const { return mState; }
    
private:
    BatteryConfig mConfig;
    BatteryState mState;
};

// Test framework
static int passed = 0, failed = 0;

#define TEST(name) void test_##name()
#define RUN(name) do { \
    std::cout << "  " << std::left << std::setw(40) << #name << ": "; \
    try { test_##name(); passed++; std::cout << "\033[32m✓ PASS\033[0m\n"; } \
    catch (const std::exception& e) { failed++; std::cout << "\033[31m✗ FAIL\033[0m - " << e.what() << "\n"; } \
} while(0)

#define ASSERT(cond, msg) if (!(cond)) throw std::runtime_error(msg)
#define ASSERT_EQ(exp, act, msg) ASSERT(std::abs((exp)-(act)) < 0.01f, std::string(msg) + " (exp:" + std::to_string(exp) + " act:" + std::to_string(act) + ")")

// Tests
TEST(battery_init_full) {
    BatterySystem b; b.Init();
    ASSERT_EQ(1000.0f, b.GetState().currentEnergy, "Initial energy");
    ASSERT_EQ(100.0f, b.GetState().GetChargePercent(), "Initial charge %");
}

TEST(wireless_charging_works) {
    BatterySystem b; b.Init();
    BatteryState& s = const_cast<BatteryState&>(b.GetState());
    s.currentEnergy = 500.0f;
    
    b.Update(1.0f, JPH::Vec3(0,0,0), JPH::Vec3(0,0,0), 1.0f);  // 1m from center
    
    ASSERT(s.currentEnergy > 500.0f, "Energy should increase");
    ASSERT(b.GetState().isCharging, "Should be charging");
}

TEST(wireless_out_of_range) {
    BatterySystem b; b.Init();
    float initial = b.GetState().currentEnergy;
    
    b.Update(1.0f, JPH::Vec3(0,0,0), JPH::Vec3(0,0,0), 10.0f);  // 10m from center
    
    ASSERT_EQ(initial, b.GetState().currentEnergy, "Energy unchanged outside range");
    ASSERT(!b.GetState().isCharging, "Should not be charging");
}

TEST(regen_braking_works) {
    BatterySystem b; b.Init();
    BatteryState& s = const_cast<BatteryState&>(b.GetState());
    s.currentEnergy = 500.0f;
    
    b.Update(0.1f, JPH::Vec3(1,0,0), JPH::Vec3(5,0,0), 100.0f);  // Decelerating
    
    ASSERT(s.currentEnergy > 500.0f, "Energy should increase from regen");
    ASSERT(b.GetState().totalRegenerated > 0.0f, "Regen should be tracked");
}

TEST(no_regen_accelerating) {
    BatterySystem b; b.Init();
    float initial = b.GetState().currentEnergy;
    
    b.Update(0.1f, JPH::Vec3(5,0,0), JPH::Vec3(1,0,0), 100.0f);  // Accelerating
    
    ASSERT_EQ(initial, b.GetState().currentEnergy, "No regen when accelerating");
}

TEST(power_request_works) {
    BatterySystem b; b.Init();
    
    float power = b.RequestPower(50.0f, PowerType::Attack, 1.0f);
    
    ASSERT_EQ(50.0f, power, "Should deliver requested power");
    ASSERT(b.GetState().isDischarging, "Should be discharging");
    ASSERT_EQ(950.0f, b.GetState().currentEnergy, "Energy should decrease");
}

TEST(power_limited_by_max) {
    BatterySystem b; b.Init();
    
    float power = b.RequestPower(150.0f, PowerType::Attack, 1.0f);
    
    ASSERT_EQ(100.0f, power, "Should be limited to maxOutput");
}

TEST(power_setting_affects_output) {
    BatterySystem b; b.Init();
    b.SetAttackPowerSetting(0.5f);
    
    float power = b.RequestPower(100.0f, PowerType::Attack, 1.0f);
    
    ASSERT_EQ(50.0f, power, "Should be limited by power setting");
}

TEST(overheat_blocks_power) {
    BatterySystem b; b.Init();
    BatteryState& s = const_cast<BatteryState&>(b.GetState());
    s.temperature = BATTERY_OVERHEAT_TEMP;
    s.isOverheated = true;
    
    float power = b.RequestPower(50.0f, PowerType::Attack, 1.0f);
    
    ASSERT_EQ(0.0f, power, "No power when overheated");
}

TEST(cooling_works) {
    BatterySystem b; b.Init();
    BatteryState& s = const_cast<BatteryState&>(b.GetState());
    s.temperature = 60.0f;
    
    b.Update(1.0f, JPH::Vec3(0,0,0), JPH::Vec3(0,0,0), 100.0f);
    
    ASSERT(s.temperature < 60.0f, "Temperature should decrease");
    ASSERT(s.temperature >= BATTERY_AMBIENT_TEMP, "Not below ambient");
}

TEST(low_power_flag) {
    BatterySystem b; b.Init();
    BatteryState& s = const_cast<BatteryState&>(b.GetState());
    
    s.currentEnergy = 150.0f;
    ASSERT(b.GetState().IsLowPower(), "Should be low at 15%");
    
    s.currentEnergy = 250.0f;
    ASSERT(!b.GetState().IsLowPower(), "Should not be low at 25%");
}

TEST(charge_discharge_tracking) {
    BatterySystem b; b.Init();
    
    // Charge
    b.Update(1.0f, JPH::Vec3(0,0,0), JPH::Vec3(0,0,0), 1.0f);
    float charged = b.GetState().totalCharged;
    
    // Discharge
    b.RequestPower(50.0f, PowerType::Attack, 1.0f);
    float discharged = b.GetState().totalDischarged;
    
    ASSERT(charged > 0.0f, "Should track charging");
    ASSERT(discharged > 0.0f, "Should track discharging");
}

int main() {
    std::cout << "\n\033[1;36m╔════════════════════════════════════════════╗\033[0m\n";
    std::cout << "\033[1;36m║   BATTERY SYSTEM STANDALONE TEST SUITE    ║\033[0m\n";
    std::cout << "\033[1;36m╚════════════════════════════════════════════╝\033[0m\n\n";
    
    std::cout << "\033[1;33m--- Core Functionality ---\033[0m\n";
    RUN(battery_init_full);
    RUN(wireless_charging_works);
    RUN(wireless_out_of_range);
    RUN(regen_braking_works);
    RUN(no_regen_accelerating);
    
    std::cout << "\n\033[1;33m--- Power Management ---\033[0m\n";
    RUN(power_request_works);
    RUN(power_limited_by_max);
    RUN(power_setting_affects_output);
    RUN(overheat_blocks_power);
    
    std::cout << "\n\033[1;33m--- Thermal & State ---\033[0m\n";
    RUN(cooling_works);
    RUN(low_power_flag);
    RUN(charge_discharge_tracking);
    
    std::cout << "\n\033[1;36m╔════════════════════════════════════════════╗\033[0m\n";
    std::cout << "  \033[1;32mPASSED: " << std::setw(3) << passed << "\033[0m  |  \033[1;31mFAILED: " << std::setw(3) << failed << "\033[0m\n";
    std::cout << "\033[1;36m╚════════════════════════════════════════════╝\033[0m\n\n";
    
    return failed > 0 ? 1 : 0;
}
