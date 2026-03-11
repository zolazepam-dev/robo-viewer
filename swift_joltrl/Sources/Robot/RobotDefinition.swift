import Foundation

public struct RobotMaterial: Codable {
    public let friction: Float
    public let restitution: Float
}

public struct RobotShape: Codable {
    public let type: String
    public let radius: Float?
}

public struct RobotBody: Codable {
    public let name: String
    public let shape: RobotShape
    public let position: [Float]
    public let mass: Float
    public let material: RobotMaterial?
}

public struct RobotConstraint: Codable {
    public let name: String
    public let type: String
    public let body1: String
    public let body2: String
    public let position: [Float]
    public let hasMotor: Bool
    public let motorMaxTorque: Float?
}

public struct RobotSensors: Codable {
    public let lidar_rays: Int
    public let lidar_max_distance: Float
}

public struct RobotActions: Codable {
    public let reaction_wheel_dim: Int
    public let reaction_torque_scale: Float
}

public struct RobotDefinition: Codable {
    public let name: String
    public let type: String
    public let bodies: [RobotBody]
    public let constraints: [RobotConstraint]
    public let sensors: RobotSensors?
    public let actions: RobotActions?
    public let useDirectTorque: Bool?
    public let orbiterTorqueScale: Float?
}

public class RobotLoader {
    public init() {}

    public func loadRobot(from path: String) throws -> RobotDefinition {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(RobotDefinition.self, from: data)
    }
}
