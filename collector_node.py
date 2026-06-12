# collector_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
import serial
import threading
import time

class RobotCollectorNode(Node):
    def __init__(self):
        super().__init__('robot_collector_node')
        
        # Declare parameters for flexibility
        self.declare_parameter('port', '/dev/ttyUSB0') # Default for Linux, change to COM5 for Windows
        self.declare_parameter('baud', 9600)
        
        port = self.get_parameter('port').get_parameter_value().string_value
        baud = self.get_parameter('baud').get_parameter_value().integer_value
        
        # ROS2 Publishers
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.current_pub = self.create_publisher(Float32, '/robot/current_draw', 10)
        self.imu_pub = self.create_publisher(Float32, '/robot/imu_vibration', 10)
        
        # Initialize Serial
        self.get_logger().info(f"Connecting to serial port {port} at {baud} baud...")
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)
            self.get_logger().info(f"Successfully connected to Arduino on {port}")
            self.serial_ok = True
        except Exception as e:
            self.get_logger().error(f"Failed to connect to Serial: {e}")
            self.serial_ok = False
            
        # Spin up a background thread to read serial data continuously
        if self.serial_ok:
            self.read_thread = threading.Thread(target=self.serial_read_loop, daemon=True)
            self.read_thread.start()

    def serial_read_loop(self):
        while rclpy.ok():
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                
                parts = line.split(',')
                if len(parts) < 6:
                    continue
                
                # Parse: base,shoulder,elbow,gripper,current,imu
                base = float(parts[0])
                shoulder = float(parts[1])
                elbow = float(parts[2])
                gripper = float(parts[3])
                current = float(parts[4])
                imu = float(parts[5])
                
                # 1. Publish Joint States (ROS expects angles in radians)
                joint_msg = JointState()
                joint_msg.header.stamp = self.get_clock().now().to_msg()
                joint_msg.name = ['base_joint', 'shoulder_joint', 'elbow_joint', 'gripper_joint']
                
                # Convert degrees to radians
                joint_msg.position = [
                    np.radians(base),
                    np.radians(shoulder),
                    np.radians(elbow),
                    np.radians(gripper)
                ]
                self.joint_pub.publish(joint_msg)
                
                # 2. Publish Current Draw
                current_msg = Float32()
                current_msg.data = current
                self.current_pub.publish(current_msg)
                
                # 3. Publish IMU Vibration metric
                imu_msg = Float32()
                imu_msg.data = imu
                self.imu_pub.publish(imu_msg)
                
            except Exception as e:
                self.get_logger().warn(f"Error parsing serial data: {e}")
                time.sleep(0.1)

    def destroy_node(self):
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
            self.get_logger().info("Closed serial port connection.")
        super().destroy_node()

# NumPy import inside ROS callback safety
import numpy as np

def main(args=None):
    rclpy.init(args=args)
    node = RobotCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
