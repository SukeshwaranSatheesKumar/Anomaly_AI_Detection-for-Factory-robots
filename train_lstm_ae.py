# train_lstm_ae.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import sys

MODEL_PATH = 'lstm_autoencoder.keras'
TRAIN_DATA = 'X_train_seq.npy'
TEST_DATA = 'X_test_seq.npy'

def build_lstm_autoencoder(window_size, num_features):
    """Defines the LSTM Autoencoder architecture."""
    model = Sequential([
        # Encoder
        LSTM(32, activation='relu', input_shape=(window_size, num_features), return_sequences=False),
        RepeatVector(window_size),
        # Decoder
        LSTM(32, activation='relu', return_sequences=True),
        TimeDistributed(Dense(num_features))
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

def main():
    # Check if preprocessed data exists
    if not os.path.exists(TRAIN_DATA):
        print(f"Error: {TRAIN_DATA} not found. Please run preprocess.py first.")
        sys.exit(1)
        
    # Load sequence data
    X_train = np.load(TRAIN_DATA)
    X_test = np.load(TEST_DATA)
    
    print(f"Loaded training sequence data: {X_train.shape}")
    print(f"Loaded test sequence data: {X_test.shape}")
    
    window_size = X_train.shape[1]
    num_features = X_train.shape[2]
    
    # Build Model
    model = build_lstm_autoencoder(window_size, num_features)
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)
    ]
    
    # Train Model
    print("Training LSTM Autoencoder model (Reconstruction based)...")
    history = model.fit(
        X_train, X_train,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate final test loss
    test_loss = model.evaluate(X_test, X_test, verbose=0)
    print(f"Test Reconstruction Loss (Mean Absolute Error): {test_loss:.4f}")
    print(f"Model saved to {MODEL_PATH}")

if __name__ == '__main__':
    main()
