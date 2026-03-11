import Foundation

public struct Transition {
    public let state: [Float]
    public let action: [Float]
    public let reward: Float
    public let nextState: [Float]
    public let done: Bool
}

public class ReplayBuffer {
    private var buffer: [Transition] = []
    private let maxSize: Int
    private var ptr: Int = 0
    
    public init(maxSize: Int = 1_000_000) {
        self.maxSize = maxSize
    }
    
    public func add(_ transition: Transition) {
        if buffer.count < maxSize {
            buffer.append(transition)
        } else {
            buffer[ptr] = transition
        }
        ptr = (ptr + 1) % maxSize
    }
    
    public func sample(batchSize: Int) -> [Transition] {
        guard buffer.count >= batchSize else { return [] }
        // Simple random sampling
        return (0..<batchSize).map { _ in buffer.randomElement()! }
    }
    
    public var count: Int { return buffer.count }
}
