# realtime_detector_ros2.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, String
import numpy as np
import joblib
from collections import deque
import os
import sys

# Model Paths
SCALER_PATH = 'scaler.joblib'
IF_MODEL_PATH = 'isolation_forest.joblib'
LSTM_MODEL_PATH = 'lstm_autoencoder.keras'
THRESHOLD_PATH = 'lstm_threshold.npy'

class RealtimeDetectorROS2(Node):
    def __init__(self):
        super().__init__('realtime_detector_ros2')
        
        # Declare parameters to choose model type
        self.declare_parameter('model_type', 'if') # 'if' (Isolation Forest) or 'lstm' (LSTM Autoencoder)
        self.model_type = self.get_parameter('model_type').get_parameter_value().string_value
        
        # 1. Load Scaler
        if not os.path.exists(SCALER_PATH):
            self.get_logger().error(f"Scaler file '{SCALER_PATH}' not found. Run preprocess.py first.")
            sys.exit(1)
        self.scaler = joblib.load(SCALER_PATH)
        
        # 2. Load Model
        self.model = None
        self.lstm_threshold = None
        
        if self.model_type == 'if':
            if not os.path.exists(IF_MODEL_PATH):
                self.get_logger().error(f"Model file '{IF_MODEL_PATH}' not found. Train first.")
                sys.exit(1)
            self.model = joblib.load(IF_MODEL_PATH)
            self.get_logger().info("ROS2 Realtime Detector: Loaded Isolation Forest Model.")
        elif self.model_type == 'lstm':
            if not os.path.exists(LSTM_MODEL_PATH):
                self.get_logger().error(f"Model file '{LSTM_MODEL_PATH}' not found. Train first.")
                sys.exit(1)
            import tensorflow as tf
            self.model = tf.keras.models.load_model(LSTM_MODEL_PATH)
            self.get_logger().info("ROS2 Realtime Detector: Loaded LSTM Autoencoder Model.")
            
            # Load threshold
            if os.path.exists(THRESHOLD_PATH):
                self.lstm_threshold = float(np.load(THRESHOLD_PATH))
                self.get_logger().info(f"Loaded LSTM Threshold: {self.lstm_threshold:.4f}")
            else:
                self.lstm_threshold = 0.1
                self.get_logger().warn("Threshold file not found. Using default fallback: 0.1")
                
        # Buffer to cache the latest asynchronous sensor values
        self.latest_joints = [90.0, 90.0, 90.0, 45.0] # Initial guess [base, shoulder, elbow, gripper] in degrees
        self.latest_current = 0.0
        self.latest_imu = 1.0
        
        # Deque for LSTM sliding window
        self.window = deque(maxlen=10)
        
        # ROS2 Subscriptions
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.current_sub = self.create_subscription(Float32, '/robot/current_draw', self.current_callback, 10)
        self.imu_sub = self.create_subscription(Float32, '/robot/imu_vibration', self.imu_callback, 10)
        
        # ROS2 Publisher for Anomaly Status
        self.status_pub = self.create_publisher(String, '/robot/anomaly_status', 10)
        
        # Timer for running inference: 20 Hz
        self.timer = self.create_timer(0.05, self.inference_timer_callback)
        self.get_logger().info("ROS2 Detector spinning. Real-time inference active.")

    def joint_callback(self, msg):
        # We look for positions of our joints
        names = msg.name
        positions = msg.position
        
        joint_map = dict(zip(names, positions))
        
        # Convert positions from radians back to degrees (as expected by training)
        try:
            self.latest_joints = [
                np.degrees(joint_map.get('base_joint', np.radians(self.latest_joints[0]))),
                np.degrees(joint_map.get('shoulder_joint', np.radians(self.latest_joints[1]))),
                np.degrees(joint_map.get('elbow_joint', np.radians(self.latest_joints[2]))),
                np.degrees(joint_map.get('gripper_joint', np.radians(self.latest_joints[3])))
            ]
        except Exception as e:
            self.get_logger().warn(f"Failed to parse joints: {e}")

    def current_callback(self, msg):
        self.latest_current = msg.data

    def imu_callback(self, msg):
        self.latest_imu = msg.data

    def inference_timer_callback(self):
        # Assemble feature vector: [base, shoulder, elbow, gripper, current, imu]
        raw_vector = self.latest_joints + [self.latest_current, self.latest_imu]
        
        # Scale input
        scaled_vector = self.scaler.transform([raw_vector])[0]
        
        status_msg = String()
        
        if self.model_type == 'if':
            prediction = self.model.predict([scaled_vector])[0]
            if prediction == -1:
                status_msg.data = "ANOMALY"
                self.status_pub.publish(status_msg)
                self.get_logger().warn(f"ANOMALY DETECTED by Isolation Forest! Current: {self.latest_current:.2f}A, IMU: {self.latest_imu:.2f}g")
            else:
                status_msg.data = "NORMAL"
                self.status_pub.publish(status_msg)
                
        elif self.model_type == 'lstm':
            self.window.append(scaled_vector)
            if len(self.window) < 10:
                return # Still filling buffer
                
            input_seq = np.array([list(self.window)])
            reconstructed = self.model.predict(input_seq, verbose=0)
            mae = np.mean(np.abs(input_seq - reconstructed))
            
            if mae > self.lstm_threshold:
                status_msg.data = f"ANOMALY: LSTM Loss ({mae:.4f}) > Thresh ({self.lstm_threshold:.4f})"
                self.status_pub.publish(status_msg)
                self.get_logger().warn(f"ANOMALY DETECTED by LSTM Autoencoder! MAE Loss: {mae:.4f}")
            else:
                status_msg.data = "NORMAL"
                self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RealtimeDetectorROS2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
