// Debug battery values with realistic game scenario
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
constexpr float SHIELD_DRAIN_RATE = 25.0f;

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
            mState.currentCharge = chargeRate;  // J/s
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
            mState.currentDraw = actual;  // J/s
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

int main() {
    std::cout << std::fixed << std::setprecision(2);
    
    std::cout << "\n╔════════════════════════════════════════════╗\n";
    std::cout << "║   BATTERY DEBUG - REALISTIC GAME VALUES   ║\n";
    std::cout << "╚════════════════════════════════════════════╝\n\n";
    
    BatterySystem battery;
    battery.Init();
    
    const float dt = 1.0f / 60.0f;  // 60 Hz physics
    
    std::cout << "Scenario: Robot at 2m from center, moving around\n";
    std::cout << "Physics timestep: " << dt << "s (60 Hz)\n\n";
    
    // Simulate 1 second of gameplay (60 frames)
    struct Frame {
        float velX, velY, velZ;
        float prevVelX, prevVelY, prevVelZ;
        float chargeDist;
        bool useEMP;
    };
    
    Frame frames[] = {
        // Frame 0-9: Moving at constant speed, 2m from center
        {3, 0, 0,  3, 0, 0,  2.0f, false},
        {3, 0, 0,  3, 0, 0,  2.0f, false},
        {3, 0, 0,  3, 0, 0,  2.0f, false},
        {3, 0, 0,  3, 0, 0,  2.0f, false},
        {3, 0, 0,  3, 0, 0,  2.0f, false},
        {3, 0, 0,  3, 0, 0,  2.0f, false},
        {3, 0, 0,  3, 0, 0,  2.0f, false},
        {3, 0, 0,  3, 0, 0,  2.0f, false},
        {3, 0, 0,  3, 0, 0,  2.0f, false},
        {3, 0, 0,  3, 0, 0,  2.0f, false},
        
        // Frame 10: BRAKING! (5 m/s -> 2 m/s)
        {2, 0, 0,  5, 0, 0,  2.0f, false},
        
        // Frame 11-19: Constant speed again
        {2, 0, 0,  2, 0, 0,  2.0f, false},
        {2, 0, 0,  2, 0, 0,  2.0f, false},
        {2, 0, 0,  2, 0, 0,  2.0f, false},
        {2, 0, 0,  2, 0, 0,  2.0f, false},
        {2, 0, 0,  2, 0, 0,  2.0f, false},
        {2, 0, 0,  2, 0, 0,  2.0f, false},
        {2, 0, 0,  2, 0, 0,  2.0f, false},
        {2, 0, 0,  2, 0, 0,  2.0f, false},
        {2, 0, 0,  2, 0, 0,  2.0f, false},
        
        // Frame 20: Use EMP ability
        {2, 0, 0,  2, 0, 0,  2.0f, true},
        
        // Frame 21-59: Normal movement
        {2, 0, 0,  2, 0, 0,  2.0f, false},
    };
    
    float totalCharge = 0;
    float totalDischarge = 0;
    float totalRegen = 0;
    
    for (int i = 0; i < 60; i++) {
        Frame& f = frames[i % 60];
        
        JPH::Vec3 vel(f.velX, f.velY, f.velZ);
        JPH::Vec3 prevVel(f.prevVelX, f.prevVelY, f.prevVelZ);
        
        battery.Update(dt, vel, prevVel, f.chargeDist);
        
        if (f.useEMP) {
            battery.RequestPower(EMP_COST, PowerType::Attack, dt);
        }
        
        const auto& s = battery.GetState();
        totalCharge += s.currentCharge * dt;
        if (s.isDischarging) totalDischarge += s.currentDraw * dt;
        
        if (i < 15 || i >= 55) {
            std::cout << "Frame " << std::setw(2) << i << ": ";
            std::cout << "Charge=" << std::setw(6) << s.currentCharge << " J/s";
            std::cout << " Draw=" << std::setw(6) << s.currentDraw << " J/s";
            std::cout << " Energy=" << std::setw(7) << s.currentEnergy << " J";
            std::cout << " Regen=" << std::setw(7) << s.totalRegenerated << " J";
            std::cout << (s.isCharging ? " [CHARGING]" : "");
            std::cout << (s.isDischarging ? " [DISCHARGING]" : "");
            if (f.useEMP) std::cout << " [EMP]";
            std::cout << "\n";
        }
    }
    
    const auto& s = battery.GetState();
    
    std::cout << "\n─────────────────────────────────────────\n";
    std::cout << "FINAL STATE (after 60 frames = 1 second):\n";
    std::cout << "  Energy:       " << s.currentEnergy << " J (started at 500 J)\n";
    std::cout << "  Net Change:   " << (s.currentEnergy - 500.0f) << " J\n";
    std::cout << "  Total Charged:    " << s.totalCharged << " J\n";
    std::cout << "  Total Discharged: " << s.totalDischarged << " J\n";
    std::cout << "  Total Regenerated:" << s.totalRegenerated << " J\n";
    std::cout << "\n";
    std::cout << "  currentCharge: " << s.currentCharge << " J/s (instantaneous)\n";
    std::cout << "  currentDraw:   " << s.currentDraw << " J/s (instantaneous)\n";
    std::cout << "\n";
    std::cout << "EXPECTED VALUES:\n";
    std::cout << "  Wireless charge at 2m: 15 * (1-2/5)^2 = 5.4 J/s\n";
    std::cout << "  Per frame (1/60s):    5.4 / 60 = 0.09 J/frame\n";
    std::cout << "  Over 60 frames:       5.4 J total\n";
    std::cout << "\n";
    std::cout << "  Regen braking (5->2 m/s): 0.5*100*(25-4)*0.3 = 315 J\n";
    std::cout << "\n";
    std::cout << "  EMP cost: 150 J * (1/60) = 2.5 J per use\n";
    std::cout << "─────────────────────────────────────────\n\n";
    
    return 0;
}
