#pragma once

#include <Jolt/Jolt.h>
#include <array>
#include <cmath>

// Battery System Configuration - Add to any robot
constexpr float BATTERY_DEFAULT_CAPACITY = 1000.0f;      // Total energy storage (Joules)
constexpr float BATTERY_DEFAULT_CHARGE_RATE = 20.0f;     // Max charge rate (J/s)
constexpr float BATTERY_DEFAULT_OUTPUT = 100.0f;         // Max output power (J/s)
constexpr float BATTERY_DEFAULT_HEAT_CAPACITY = 100.0f;  // Heat storage before overheat
constexpr float BATTERY_DEFAULT_COOLING = 5.0f;          // Passive cooling (heat/s)
constexpr float BATTERY_AMBIENT_TEMP = 20.0f;            // Ambient temperature (Celsius)
constexpr float BATTERY_OVERHEAT_TEMP = 80.0f;           // Overheat cutoff (Celsius)
constexpr float BATTERY_REGEN_EFFICIENCY = 0.3f;         // 30% energy recovery from braking
constexpr float WIRELESS_CHARGE_RANGE = 5.0f;            // Max wireless charging range (meters)
constexpr float WIRELESS_CHARGE_MAX = 15.0f;             // Max wireless charge rate at 0 distance

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
    // Energy
    float currentEnergy = BATTERY_DEFAULT_CAPACITY;
    float totalCharged = 0.0f;
    float totalDischarged = 0.0f;
    float totalRegenerated = 0.0f;
    
    // Thermal
    float temperature = BATTERY_AMBIENT_TEMP;
    float totalHeatGenerated = 0.0f;
    float totalHeatDissipated = 0.0f;
    
    // Status
    bool isOverheated = false;
    bool isCharging = false;
    bool isDischarging = false;
    float currentDraw = 0.0f;
    float currentCharge = 0.0f;
    
    // Output settings (0.0 to 1.0)
    float attackPowerSetting = 1.0f;
    float movementPowerSetting = 1.0f;
    float shieldPowerSetting = 1.0f;
    
    void Reset(const BatteryConfig& config) {
        currentEnergy = config.capacity;
        temperature = config.ambientTemp;
        isOverheated = false;
        isCharging = false;
        isDischarging = false;
        currentDraw = 0.0f;
        currentCharge = 0.0f;
        totalCharged = 0.0f;
        totalDischarged = 0.0f;
        totalRegenerated = 0.0f;
        totalHeatGenerated = 0.0f;
        totalHeatDissipated = 0.0f;
        attackPowerSetting = 1.0f;
        movementPowerSetting = 1.0f;
        shieldPowerSetting = 1.0f;
    }
    
    float GetChargePercent() const { return currentEnergy / BATTERY_DEFAULT_CAPACITY * 100.0f; }
    float GetHeatPercent() const { return (temperature - BATTERY_AMBIENT_TEMP) / (BATTERY_OVERHEAT_TEMP - BATTERY_AMBIENT_TEMP) * 100.0f; }
    bool IsLowPower() const { return currentEnergy < (BATTERY_DEFAULT_CAPACITY * 0.2f); }
    bool IsCriticalPower() const { return currentEnergy < (BATTERY_DEFAULT_CAPACITY * 0.1f); }
};

class BatterySystem {
public:
    BatterySystem() { Init(); }
    
    void Init(const BatteryConfig& config = BatteryConfig()) {
        mConfig = config;
        mState.Reset(config);
    }
    
    void Update(float dt, const JPH::Vec3& velocity, const JPH::Vec3& prevVelocity, 
                float wirelessChargeDistance = 1000.0f) {
        // Reset flags
        mState.isCharging = false;
        mState.isDischarging = false;
        mState.currentDraw = 0.0f;
        mState.currentCharge = 0.0f;
        
        // 1. Wireless charging (distance-based)
        if (wirelessChargeDistance < mConfig.wirelessRange && !mState.isOverheated) {
            float chargeFactor = 1.0f - (wirelessChargeDistance / mConfig.wirelessRange);
            float chargeRate = mConfig.wirelessMaxRate * chargeFactor * chargeFactor; // Falloff
            float heatGen = chargeRate * 0.1f; // Charging generates some heat
            
            mState.currentEnergy = fminf(mState.currentEnergy + chargeRate * dt, mConfig.capacity);
            mState.currentCharge = chargeRate;
            mState.isCharging = true;
            mState.totalCharged += chargeRate * dt;
            mState.totalHeatGenerated += heatGen * dt;
            mState.temperature += heatGen * dt / mConfig.heatCapacity;
        }
        
        // 2. Regenerative braking (any deceleration)
        float currentSpeed = velocity.Length();
        float prevSpeed = prevVelocity.Length();
        if (currentSpeed < prevSpeed && currentSpeed > 0.1f && !mState.isOverheated) {
            float kineticEnergy = 0.5f * 100.0f * (prevSpeed * prevSpeed - currentSpeed * currentSpeed); // Assume 100kg mass
            float recovered = kineticEnergy * mConfig.regenEfficiency;
            
            mState.currentEnergy = fminf(mState.currentEnergy + recovered, mConfig.capacity);
            mState.totalRegenerated += recovered;
            mState.totalHeatGenerated += recovered * 0.2f; // Regen generates heat
        }
        
        // 3. Passive cooling
        float tempDiff = mState.temperature - mConfig.ambientTemp;
        float heatDissipated = mConfig.coolingRate * tempDiff * dt;
        mState.temperature = fmaxf(mConfig.ambientTemp, mState.temperature - heatDissipated / mConfig.heatCapacity);
        mState.totalHeatDissipated += heatDissipated;
        
        // 4. Overheat check
        if (mState.temperature >= mConfig.overheatTemp) {
            mState.isOverheated = true;
        }
        
        // 5. Auto-recover from overheat when cooled
        if (mState.isOverheated && mState.temperature < (mConfig.overheatTemp - 10.0f)) {
            mState.isOverheated = false;
        }
    }
    
    // Request power - returns actual power available
    float RequestPower(float requested, PowerType type, float dt) {
        if (mState.isOverheated) return 0.0f;
        
        // Apply power setting based on type
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
            
            // Heat generation from discharge
            float heatGen = actual * 0.15f; // 15% inefficiency
            mState.totalHeatGenerated += heatGen * dt;
            mState.temperature += heatGen * dt / mConfig.heatCapacity;
        }
        
        return actual;
    }
    
    // Set power output limits (0.0 to 1.0)
    void SetAttackPowerSetting(float setting) { 
        mState.attackPowerSetting = fmaxf(0.1f, fminf(1.0f, setting)); 
    }
    void SetMovementPowerSetting(float setting) { 
        mState.movementPowerSetting = fmaxf(0.1f, fminf(1.0f, setting)); 
    }
    void SetShieldPowerSetting(float setting) { 
        mState.shieldPowerSetting = fmaxf(0.1f, fminf(1.0f, setting)); 
    }
    
    const BatteryState& GetState() const { return mState; }
    const BatteryConfig& GetConfig() const { return mConfig; }
    
private:
    BatteryConfig mConfig;
    BatteryState mState;
};
