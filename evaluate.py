# evaluate.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import sys

# Disable GPU for evaluation to avoid TF warnings if not configured
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

IF_MODEL_PATH = 'isolation_forest.joblib'
LSTM_MODEL_PATH = 'lstm_autoencoder.keras'
SCALER_PATH = 'scaler.joblib'

def main():
    # Check if files exist
    if not (os.path.exists(IF_MODEL_PATH) or os.path.exists(LSTM_MODEL_PATH)):
        print("Error: Models not found. Please train models first using train_isolation.py or train_lstm_ae.py.")
        sys.exit(1)
        
    print("Loading models and test datasets...")
    scaler = joblib.load(SCALER_PATH)
    
    # Setup plotting
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle("Anomaly AI Detection - Model Evaluations", fontsize=16, fontweight='bold')
    
    # ------------------
    # 1. Evaluate Isolation Forest
    # ------------------
    if os.path.exists(IF_MODEL_PATH) and os.path.exists('X_test_tabular.npy'):
        clf = joblib.load(IF_MODEL_PATH)
        X_test_tab = np.load('X_test_tabular.npy')
        
        # Predict anomalies (-1 = anomaly, 1 = normal)
        preds = clf.predict(X_test_tab)
        scores = clf.decision_function(X_test_tab)
        
        # Convert to unscaled data for plotting
        X_test_unscaled = scaler.inverse_transform(X_test_tab)
        
        # Plot ACS712 Current and highlight anomalies
        ax1 = axes[0]
        indices = np.arange(len(X_test_unscaled))
        
        ax1.plot(indices, X_test_unscaled[:, 4], label="Current Draw (Amps)", color='royalblue', alpha=0.8)
        
        # Find indices of anomalies
        anomaly_idx = np.where(preds == -1)[0]
        ax1.scatter(anomaly_idx, X_test_unscaled[anomaly_idx, 4], color='crimson', label='Detected Anomaly', s=30, zorder=5)
        
        ax1.set_title("Isolation Forest Anomaly Detection (Current Sensor Profile)", fontsize=12)
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Current (A)")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        print(f"[Isolation Forest] Evaluated. Detected {len(anomaly_idx)} anomalous states out of {len(X_test_tab)} steps.")
    else:
        axes[0].text(0.5, 0.5, "Isolation Forest model/data missing", ha='center', va='center')
        
    # ------------------
    # 2. Evaluate LSTM Autoencoder
    # ------------------
    if os.path.exists(LSTM_MODEL_PATH) and os.path.exists('X_test_seq.npy'):
        import tensorflow as tf
        model = tf.keras.models.load_model(LSTM_MODEL_PATH)
        X_test_seq = np.load('X_test_seq.npy')
        
        # Reconstruct inputs
        X_pred = model.predict(X_test_seq)
        
        # Calculate mean absolute error (MAE) for each window
        # MAE shape will be [samples, window_size, features] -> reduce mean over window and features
        mae_loss = np.mean(np.abs(X_pred - X_test_seq), axis=(1, 2))
        
        # Define anomaly threshold (e.g. 98th percentile of reconstruction loss)
        threshold = np.percentile(mae_loss, 95)
        
        # Classify anomalies
        anomalies = mae_loss > threshold
        num_anomalies = np.sum(anomalies)
        
        # Plot reconstruction error
        ax2 = axes[1]
        indices_seq = np.arange(len(mae_loss))
        
        ax2.plot(indices_seq, mae_loss, label="Reconstruction Loss (MAE)", color='darkorchid')
        ax2.axhline(threshold, color='crimson', linestyle='--', linewidth=2, label=f'Anomaly Threshold ({threshold:.4f})')
        
        # Highlight anomalies
        anomaly_seq_idx = np.where(anomalies)[0]
        ax2.scatter(anomaly_seq_idx, mae_loss[anomaly_seq_idx], color='crimson', label='Detected Anomaly', s=30, zorder=5)
        
        ax2.set_title("LSTM Autoencoder Anomaly Detection (Reconstruction Error)", fontsize=12)
        ax2.set_xlabel("Time Step (Windows)")
        ax2.set_ylabel("Loss (MAE)")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        # Save threshold value to file for real-time detector usage
        np.save('lstm_threshold.npy', threshold)
        print(f"[LSTM Autoencoder] Evaluated. Saved threshold {threshold:.4f}. Detected {num_anomalies} anomalous sequences.")
    else:
        axes[1].text(0.5, 0.5, "LSTM Autoencoder model/data missing", ha='center', va='center')
        
    plt.tight_layout()
    plot_file = 'evaluation_plot.png'
    plt.savefig(plot_file, dpi=300)
    plt.close()
    
    print(f"Evaluation report figures saved successfully to '{plot_file}'!")

if __name__ == '__main__':
    main()
