import Foundation
import Physics
import Robot
import Environment
import Training

print("--- Swift JOLTrl Training Pipeline ---")

// 1. Load Robot Definition
let robotPath = "robots/bouncy_orbiter.json"
print("[Main] Loading robot from \(robotPath)...")
let loader = RobotLoader()
guard let robotDef = try? loader.loadRobot(from: robotPath) else {
    print("Error: Could not load robot definition.")
    exit(1)
}
print("Loaded robot: \(robotDef.name) with \(robotDef.bodies.count) bodies.")

// 2. Initialize Physics
let physicsWorld = PhysicsWorld()
if physicsWorld.initPhysics(numParallelEnvs: 128) {
    print("Physics system ready with \(physicsWorld.getNumEnvs()) environments.")
}

// 3. Initialize Environment
let env = CombatEnv(physics: physicsWorld, robotDef: robotDef, maxSteps: 1000)

// 4. Initialize Trainer (TD3)
let stateDim = 256
let actionDim = 56 // Simplified, real dim depends on robot joints
let trainer = TD3Trainer(stateDim: stateDim, actionDim: actionDim)
print("TD3 Trainer initialized.")

// 5. Training Loop
let totalEpisodes = 100
print("\nStarting training for \(totalEpisodes) episodes...")

for episode in 0..<totalEpisodes {
    env.resetAll()
    var done = false
    var totalReward: Float = 0.0
    var steps = 0
    
    // We need initial observation - for now mocked in step()
    // In real loop: obs = env.reset()
    var obs = Array(repeating: Float(0.0), count: stateDim)
    
    while !done && steps < 1000 {
        // Select Action
        let action = trainer.selectAction(state: obs)
        
        // Step Environment (broadcasting single action for demo simplicity, or batching)
        // Here we pretend we run 1 env for the trainer logic, but sim runs 128
        let results = env.step(actions: [action]) 
        let result = results[0] // Take first env result
        
        // Store Experience
        trainer.addExperience(state: obs, action: action, reward: result.reward, nextState: result.observation, done: result.done)
        
        // Train
        trainer.trainStep()
        
        // Update State
        obs = result.observation
        totalReward += result.reward
        done = result.done
        steps += 1
    }
    
    if episode % 10 == 0 {
        print("Episode \(episode): Reward = \(String(format: "%.2f", totalReward)), Steps = \(steps)")
    }
}

print("\n--- Training Complete ---")
physicsWorld.shutdown()
