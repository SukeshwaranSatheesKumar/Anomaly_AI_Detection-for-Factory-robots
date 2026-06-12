# realtime_detector_serial.py
import serial
import time
import numpy as np
import joblib
from collections import deque
import os
import sys
import argparse

# Configuration
DEFAULT_PORT = 'COM5'
BAUD_RATE = 9600
WINDOW_SIZE = 10
FEATURES = ['base', 'shoulder', 'elbow', 'gripper', 'current', 'imu']

def main():
    parser = argparse.ArgumentParser(description="Realtime Robot Anomaly AI Detector")
    parser.add_argument('--port', type=str, default=DEFAULT_PORT, help="Serial port (e.g. COM5 or /dev/ttyUSB0)")
    parser.add_argument('--model', type=str, default='if', choices=['if', 'lstm'], help="Model type: 'if' (Isolation Forest) or 'lstm' (LSTM Autoencoder)")
    args = parser.parse_args()

    # 1. Load Scaler
    if not os.path.exists('scaler.joblib'):
        print("Error: scaler.joblib not found. Please run preprocess.py and train first.")
        sys.exit(1)
    
    scaler = joblib.load('scaler.joblib')
    print("StandardScaler loaded.")

    # 2. Load Selected AI Model
    model = None
    threshold = None
    
    if args.model == 'if':
        if not os.path.exists('isolation_forest.joblib'):
            print("Error: isolation_forest.joblib not found. Please train it first.")
            sys.exit(1)
        model = joblib.load('isolation_forest.joblib')
        print("Isolation Forest model loaded successfully.")
    elif args.model == 'lstm':
        if not os.path.exists('lstm_autoencoder.keras'):
            print("Error: lstm_autoencoder.keras not found. Please train it first.")
            sys.exit(1)
        # Load tensorflow dynamically to speed up script launch if using Isolation Forest
        import tensorflow as tf
        model = tf.keras.models.load_model('lstm_autoencoder.keras')
        print("LSTM Autoencoder model loaded successfully.")
        
        # Load threshold
        if os.path.exists('lstm_threshold.npy'):
            threshold = np.load('lstm_threshold.npy')
            print(f"LSTM reconstruction error threshold: {threshold:.4f}")
        else:
            threshold = 0.1 # Fallback default
            print(f"Warning: lstm_threshold.npy not found. Using default fallback: {threshold}")

    # 3. Connect to Serial Port
    print(f"Connecting to Arduino on {args.port} at {BAUD_RATE} baud...")
    try:
        ser = serial.Serial(args.port, BAUD_RATE, timeout=1)
        time.sleep(2) # Wait for serial connection to stabilize
        print(f"Connected to {args.port}. Monitoring robot telemetry in real-time...")
    except Exception as e:
        print(f"Failed to connect to serial port {args.port}: {e}")
        print("Please check connection and port name.")
        sys.exit(1)

    # Deque to hold sliding window of scaled data (needed for LSTM)
    window = deque(maxlen=WINDOW_SIZE)

    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            
            # Parse CSV values
            parts = line.split(',')
            if len(parts) < 6:
                continue
            
            try:
                # Format: base,shoulder,elbow,gripper,current,imu
                raw_values = [float(val) for val in parts[:6]]
            except ValueError:
                # Handle conversion error
                continue
            
            # Print raw stream
            base, shoulder, elbow, gripper, current, imu = raw_values
            print(f"Telemetry -> Joint Angles: [{int(base)}, {int(shoulder)}, {int(elbow)}, {int(gripper)}] | Current: {current:.2f}A | IMU: {imu:.2f}g", end=' ')

            # Preprocess & Scale
            # scaler expects 2D array: [1, num_features]
            scaled_values = scaler.transform([raw_values])[0]
            
            # Model Inference
            if args.model == 'if':
                # Isolation Forest predicts per sample
                prediction = model.predict([scaled_values])[0]
                
                # Check for anomaly
                if prediction == -1:
                    print(" \033[91m[ALERT: ANOMALY DETECTED (Isolation Forest)]\033[0m")
                else:
                    print(" \033[92m[OK]\033[0m")
                    
            elif args.model == 'lstm':
                window.append(scaled_values)
                
                # We need a full window of data before we can perform LSTM inference
                if len(window) < WINDOW_SIZE:
                    print(" [Filling buffer...]")
                    continue
                
                # Convert window deque to numpy array of shape [1, window_size, features]
                input_seq = np.array([list(window)])
                
                # Reconstruct
                reconstructed = model.predict(input_seq, verbose=0)
                
                # Compute Mean Absolute Error
                mae = np.mean(np.abs(input_seq - reconstructed))
                
                if mae > threshold:
                    print(f" \033[91m[ALERT: ANOMALY DETECTED (LSTM MAE: {mae:.4f} > Thresh: {threshold:.4f})]\033[0m")
                else:
                    print(f" \033[92m[OK (MAE: {mae:.4f})]\033[0m")

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    finally:
        ser.close()
        print("Serial connection closed.")

if __name__ == '__main__':
    main()
