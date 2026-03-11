import Foundation
import Neural

public class TD3Trainer {
    private let actor: SpanNetwork
    private let actorTarget: SpanNetwork
    private let critic1: SpanNetwork
    private let critic1Target: SpanNetwork
    private let critic2: SpanNetwork
    private let critic2Target: SpanNetwork
    
    private let replayBuffer: ReplayBuffer
    private let batchSize: Int
    private let gamma: Float
    private let tau: Float
    private let noise: Float
    
    public init(stateDim: Int, actionDim: Int, batchSize: Int = 256, gamma: Float = 0.99, tau: Float = 0.005, noise: Float = 0.1) {
        self.batchSize = batchSize
        self.gamma = gamma
        self.tau = tau
        self.noise = noise
        self.replayBuffer = ReplayBuffer()
        
        let actorConfig = [
            SpanLayerConfig(inputDim: stateDim, outputDim: 256),
            SpanLayerConfig(inputDim: 256, outputDim: 256),
            SpanLayerConfig(inputDim: 256, outputDim: actionDim)
        ]
        
        // Critic takes (State + Action)
        let criticConfig = [
            SpanLayerConfig(inputDim: stateDim + actionDim, outputDim: 256),
            SpanLayerConfig(inputDim: 256, outputDim: 256),
            SpanLayerConfig(inputDim: 256, outputDim: 1)
        ]
        
        self.actor = SpanNetwork()
        self.actor.initNetwork(configs: actorConfig)
        self.actorTarget = SpanNetwork()
        self.actorTarget.initNetwork(configs: actorConfig)
        self.actorTarget.copyWeights(from: self.actor)
        
        self.critic1 = SpanNetwork()
        self.critic1.initNetwork(configs: criticConfig)
        self.critic1Target = SpanNetwork()
        self.critic1Target.initNetwork(configs: criticConfig)
        self.critic1Target.copyWeights(from: self.critic1)
        
        self.critic2 = SpanNetwork()
        self.critic2.initNetwork(configs: criticConfig)
        self.critic2Target = SpanNetwork()
        self.critic2Target.initNetwork(configs: criticConfig)
        self.critic2Target.copyWeights(from: self.critic2)
    }
    
    public func selectAction(state: [Float], addNoise: Bool = true) -> [Float] {
        var action = actor.forward(input: state)
        if addNoise {
            // Add exploration noise
            action = action.map { $0 + Float.random(in: -noise...noise) }
        }
        // Clip to [-1, 1]
        return action.map { min(max($0, -1.0), 1.0) }
    }
    
    public func addExperience(state: [Float], action: [Float], reward: Float, nextState: [Float], done: Bool) {
        let transition = Transition(
            state: state,
            action: action,
            reward: reward,
            nextState: nextState,
            done: done
        )
        replayBuffer.add(transition)
    }
    
    public func trainStep() {
        if replayBuffer.count < batchSize { return }
        
        // let batch = replayBuffer.sample(batchSize: batchSize)
        
        // 1. Update Critic
        // target_actions = actor_target(next_states) + noise
        // target_Q = min(critic1_target(next_states, target_actions), critic2_target(...))
        // target_Q = reward + (1 - done) * gamma * target_Q
        // current_Q1 = critic1(states, actions)
        // loss = MSE(current_Q1, target_Q) + MSE(current_Q2, target_Q)
        // optimize(critic1, critic2)
        
        // 2. Update Actor (delayed)
        // loss = -critic1(states, actor(states))
        // optimize(actor)
        
        // 3. Update Targets
        // update_target(actor, tau)
        // update_target(critic1, tau)
        // update_target(critic2, tau)
        
        // print("[Training] Performed TD3 update step.") 
    }
}
