# fake_joint_publisher.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
import numpy as np

class FakeJointPublisher(Node):
    def __init__(self):
        super().__init__('fake_joint_publisher')
        
        # ROS2 Publishers
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.current_pub = self.create_publisher(Float32, '/robot/current_draw', 10)
        self.imu_pub = self.create_publisher(Float32, '/robot/imu_vibration', 10)
        
        # Publish timer: 20 Hz (every 50ms)
        self.timer = self.create_timer(0.05, self.publish_telemetry)
        self.count = 0.0
        
        self.get_logger().info("ROS2 Fake Joint Publisher initialized. Publishing synthetic robot telemetry...")

    def publish_telemetry(self):
        self.count += 0.05
        
        # 1. Simulate joints sweeping back and forth (radians)
        base_pos = np.radians(90 + 60 * np.sin(self.count))
        shoulder_pos = np.radians(90 + 30 * np.cos(self.count * 0.5))
        elbow_pos = np.radians(90 + 40 * np.sin(self.count * 0.8))
        gripper_pos = np.radians(45 + 15 * np.cos(self.count * 1.2))
        
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = ['base_joint', 'shoulder_joint', 'elbow_joint', 'gripper_joint']
        joint_msg.position = [base_pos, shoulder_pos, elbow_pos, gripper_pos]
        self.joint_pub.publish(joint_msg)
        
        # 2. Simulate standard current draw (Amps) with minor random electrical noise
        # Base velocity derivative correlates to current draw
        vel_base = np.abs(60 * np.cos(self.count)) * 0.02
        current_val = 0.15 + vel_base + np.random.normal(0, 0.01)
        
        # 3. Simulate IMU vibration (G's)
        imu_val = 1.0 + np.random.normal(0, 0.03)
        
        # Inject periodic mock anomaly (jam) every 30 seconds for 2 seconds
        # Period = 30s, Anomaly duration = 2s
        cycle_time = self.count % 30.0
        if 15.0 < cycle_time < 17.0:
            # Simulate a joint freeze & motor stall (current spike + IMU vibration spike)
            current_val = 2.2 + np.random.normal(0, 0.05)
            imu_val = 1.8 + np.random.normal(0, 0.1)
            # Override joint positions to simulate locking up
            joint_msg.position = [
                np.radians(90 + 60 * np.sin(15.0)),
                np.radians(90 + 30 * np.cos(15.0 * 0.5)),
                np.radians(90 + 40 * np.sin(15.0 * 0.8)),
                np.radians(45 + 15 * np.cos(15.0 * 1.2))
            ]
            self.joint_pub.publish(joint_msg)
            if int(self.count * 2) % 2 == 0:
                self.get_logger().info("Simulating Robot Jam Anomaly (Mocking active)...")
        
        # Publish current and IMU
        current_msg = Float32()
        current_msg.data = float(current_val)
        self.current_pub.publish(current_msg)
        
        imu_msg = Float32()
        imu_msg.data = float(imu_val)
        self.imu_pub.publish(imu_msg)

def main(args=None):
    rclpy.init(args=args)
    node = FakeJointPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
