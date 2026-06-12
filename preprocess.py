# preprocess.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Config
WINDOW_SIZE = 10  # Number of lookback timesteps for LSTM
STEP_SIZE = 1     # Sliding window step
SCALER_PATH = 'scaler.joblib'
DATA_PATH = 'telemetry.csv'

def generate_dummy_data():
    """Generates synthetic telemetry data for testing and demonstration."""
    print("No telemetry.csv found. Generating synthetic telemetry data...")
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate normal movement patterns (sine waves)
    t = np.linspace(0, 50, n_samples)
    base = 90 + 60 * np.sin(t)
    shoulder = 90 + 30 * np.cos(t * 0.5)
    elbow = 90 + 40 * np.sin(t * 0.8)
    gripper = 45 + 15 * np.cos(t * 1.2)
    
    # Normal current is low with small spikes at motion shifts
    current = 0.2 + 0.1 * np.abs(np.diff(base, prepend=90)) + np.random.normal(0, 0.02, n_samples)
    
    # Normal IMU magnitude (about 1g gravity + minor noise)
    imu = 1.0 + np.random.normal(0, 0.05, n_samples)
    
    # Inject some anomalies in the second half
    # Anomaly 1: Current spike (jam) around index 600-630
    current[600:630] += 1.8 + np.random.normal(0, 0.1, 30)
    
    # Anomaly 2: Extreme vibration (IMU) around index 800-840
    imu[800:840] += np.random.normal(0.8, 0.5, 40)
    
    # Save to csv
    df = pd.DataFrame({
        't': np.arange(n_samples),
        'base': base,
        'shoulder': shoulder,
        'elbow': elbow,
        'gripper': gripper,
        'current': current,
        'imu': imu
    })
    df.to_csv(DATA_PATH, index=False)
    print(f"Synthetic data saved to {DATA_PATH} ({n_samples} samples, anomalies injected at 600-630 and 800-840).")

def create_windows(data, window_size, step_size):
    """Creates overlapping sliding windows for LSTM input."""
    X = []
    for i in range(0, len(data) - window_size, step_size):
        X.append(data[i:(i + window_size)])
    return np.array(X)

def main():
    if not os.path.exists(DATA_PATH):
        generate_dummy_data()
        
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    
    # Drop timestamp column
    features = ['base', 'shoulder', 'elbow', 'gripper', 'current', 'imu']
    data = df[features].values
    
    # 1. Scale Features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Save the scaler for use in evaluation and real-time inference
    joblib.dump(scaler, SCALER_PATH)
    print(f"StandardScaler saved to {SCALER_PATH}")
    
    # 2. Prepare Tabular Data (for Isolation Forest)
    # We split tabular scaled rows directly
    X_train_tab, X_test_tab = train_test_split(data_scaled, test_size=0.2, random_state=42, shuffle=False)
    
    np.save('X_train_tabular.npy', X_train_tab)
    np.save('X_test_tabular.npy', X_test_tab)
    print(f"Tabular datasets saved: X_train_tabular.npy {X_train_tab.shape}, X_test_tabular.npy {X_test_tab.shape}")
    
    # 3. Prepare Sequence Data (for LSTM Autoencoder)
    data_seq = create_windows(data_scaled, WINDOW_SIZE, STEP_SIZE)
    
    # Split sequences
    X_train_seq, X_test_seq = train_test_split(data_seq, test_size=0.2, random_state=42, shuffle=False)
    
    np.save('X_train_seq.npy', X_train_seq)
    np.save('X_test_seq.npy', X_test_seq)
    print(f"Sequence datasets saved: X_train_seq.npy {X_train_seq.shape}, X_test_seq.npy {X_test_seq.shape}")
    print("Preprocessing completed successfully!")

if __name__ == '__main__':
    main()
