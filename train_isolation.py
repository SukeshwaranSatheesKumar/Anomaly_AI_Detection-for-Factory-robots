# train_isolation.py
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os
import sys

MODEL_PATH = 'isolation_forest.joblib'
TRAIN_DATA = 'X_train_tabular.npy'
TEST_DATA = 'X_test_tabular.npy'

def main():
    # Check if preprocessed data exists
    if not os.path.exists(TRAIN_DATA):
        print(f"Error: {TRAIN_DATA} not found. Please run preprocess.py first.")
        sys.exit(1)
        
    # Load training and test tabular data
    X_train = np.load(TRAIN_DATA)
    X_test = np.load(TEST_DATA)
    
    print(f"Loaded training data: {X_train.shape}")
    print(f"Loaded test data: {X_test.shape}")
    
    # Initialize Isolation Forest
    # contamination defines the proportion of outliers in the data set (e.g. 5%)
    clf = IsolationForest(
        n_estimators=150,
        max_samples='auto',
        contamination=0.05, 
        random_state=42,
        n_jobs=-1
    )
    
    print("Training Isolation Forest model...")
    clf.fit(X_train)
    
    # Evaluate predictions
    # -1 represents an anomaly, 1 represents a normal sample
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    
    train_anomalies = np.sum(y_pred_train == -1)
    test_anomalies = np.sum(y_pred_test == -1)
    
    print(f"Training Anomaly Detection Rate: {train_anomalies / len(X_train) * 100:.2f}% ({train_anomalies}/{len(X_train)} samples)")
    print(f"Test Anomaly Detection Rate: {test_anomalies / len(X_test) * 100:.2f}% ({test_anomalies}/{len(X_test)} samples)")
    
    # Save the model
    joblib.dump(clf, MODEL_PATH)
    print(f"Model successfully saved to {MODEL_PATH}")

if __name__ == '__main__':
    main()
