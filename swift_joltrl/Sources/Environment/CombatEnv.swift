import Foundation
import Physics
import Robot

public struct StepResult {
    public let observation: [Float]
    public let reward: Float
    public let done: Bool
    public let info: [String: Any]
}

public class CombatEnv {
    private let physics: PhysicsWorld
    private let robotDef: RobotDefinition
    private let numEnvs: Int
    private let maxSteps: Int
    
    // State tracking
    private var currentSteps: [Int]
    private var robotBodyIDs: [[Int]] = [] // [envIdx][robotIdx * bodies + bodyIdx]
    private var scores: [Float]

    public init(physics: PhysicsWorld, robotDef: RobotDefinition, maxSteps: Int = 1000) {
        self.physics = physics
        self.robotDef = robotDef
        self.numEnvs = Int(physics.getNumEnvs())
        self.maxSteps = maxSteps
        
        self.currentSteps = Array(repeating: 0, count: numEnvs)
        self.scores = Array(repeating: 0.0, count: numEnvs)
        
        resetAll()
    }
    
    public func resetAll() {
        // Clear old bodies (mock logic)
        robotBodyIDs = []
        
        for envIdx in 0..<numEnvs {
            var envBodyIDs: [Int] = []
            
            // Spawn 2 robots for 1v1 combat
            // Robot 1 (Agent) at (-5, 0, 0)
            envBodyIDs.append(contentsOf: spawnRobot(at: [-5.0, 5.0, 0.0], envIdx: envIdx))
            
            // Robot 2 (Opponent) at (5, 0, 0)
            envBodyIDs.append(contentsOf: spawnRobot(at: [5.0, 5.0, 0.0], envIdx: envIdx))
            
            robotBodyIDs.append(envBodyIDs)
            currentSteps[envIdx] = 0
            scores[envIdx] = 0.0
        }
        print("[CombatEnv] Reset \(numEnvs) environments with 1v1 setup.")
    }
    
    private func spawnRobot(at position: [Float], envIdx: Int) -> [Int] {
        var ids: [Int] = []
        // Map body names to IDs for constraint creation
        var bodyMap: [String: Int] = [:]
        
        // 1. Create Bodies
        for body in robotDef.bodies {
            let worldPos = [
                position[0] + body.position[0],
                position[1] + body.position[1],
                position[2] + body.position[2]
            ]
            
            let id = physics.createBody(
                name: body.name,
                shapeType: body.shape.type,
                radius: body.shape.radius ?? 0.5,
                position: worldPos,
                mass: body.mass,
                friction: body.material?.friction ?? 0.5,
                restitution: body.material?.restitution ?? 0.0
            )
            ids.append(id)
            bodyMap[body.name] = id
        }
        
        // 2. Create Constraints
        for constraint in robotDef.constraints {
            if let b1 = bodyMap[constraint.body1], let b2 = bodyMap[constraint.body2] {
                physics.createConstraint(
                    name: constraint.name,
                    type: constraint.type,
                    body1: b1,
                    body2: b2,
                    pivot: constraint.position, // Relative to body1 usually, simplified here
                    motorTorque: constraint.motorMaxTorque
                )
            }
        }
        return ids
    }
    
    public func step(actions: [[Float]]) -> [StepResult] {
        // 1. Apply Actions (Torques/Forces)
        // In real implementation: ApplyTorque(bodyID, action[i])
        
        // 2. Step Physics
        physics.step(deltaTime: 1.0 / 60.0)
        
        // 3. Compute Observations & Rewards
        var results: [StepResult] = []
        
        for i in 0..<numEnvs {
            currentSteps[i] += 1
            
            // Fake observation: 256 floats
            let obs = (0..<256).map { _ in Float.random(in: -1...1) }
            
            // Fake reward: based on "damage" (random for now)
            let reward = Float.random(in: -0.1...0.1)
            scores[i] += reward
            
            let done = currentSteps[i] >= maxSteps
            
            results.append(StepResult(
                observation: obs,
                reward: reward,
                done: done,
                info: ["score": scores[i]]
            ))
            
            if done {
                resetEnv(i)
            }
        }
        return results
    }
    
    private func resetEnv(_ index: Int) {
        currentSteps[index] = 0
        scores[index] = 0.0
        // In real sim: Reset positions of bodies for this env
    }
}
