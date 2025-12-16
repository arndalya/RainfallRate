import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

def create_sequences(X, y, time_steps=2):
    """Fungsi buat sequence dari kode teman - sama persis"""
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i : i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

def train(X, y, lr, epochs, mode="Training"):
    """
    Fungsi utama yang dipanggil dari app.py
    Input: X (features), y (target RR), lr, epochs, mode
    Output: predictions, errors_list, model_info
    """
    
    # Mapping kolom app.py ke kolom LSTM teman
    feature_map = {'Tx': 'TX', 'Tn': 'TN', 'RH_avg': 'RH_AVG', 'ss': 'SS'}
    # X_renamed = X.rename(columns=feature_map)
    
    # 1. Normalisasi data (tambahan untuk konsistensi)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    # X_scaled = scaler_X.fit_transform(X_renamed)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
    
    # 2. Buat sequences
    timesteps = 2
    n_features = X_scaled.shape[1]
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, timesteps)
    
    # 3. Bangun model LSTM (sama persis)
    model = Sequential()
    model.add(LSTM(60, input_shape=(timesteps, n_features), return_sequences=True))
    model.add(LSTM(60))
    model.add(Dense(1))  # n_features_output = 1 untuk RR
    
    # Custom learning rate dari Streamlit
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    
    # 4. Training dengan early stopping
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_seq, y_seq,
        epochs=epochs,
        batch_size=32,
        verbose=0,  # Silent untuk Streamlit
        callbacks=[early_stop]
    )
    
    # Simpan semua info model
    model_info = {
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'history': history.history,
        'timesteps': timesteps
    }
    
    os.makedirs("file.npz", exist_ok=True)

    # model.save("file.npz/lstm_model.h5")
    joblib.dump(scaler_X, "file.npz/scaler_X.pkl")
    joblib.dump(scaler_y, "file.npz/scaler_y.pkl")

    # 5. Hitung hasil berdasarkan MODE
    loss_history = np.array(history.history.get('loss', []))
    rmse_history = np.sqrt(loss_history).tolist()  # RMSE dari MSE sebagai list
    
    if mode == "Training":
        # Error training dari history
        predictions_scaled = model.predict(X_seq, verbose=0)
        predictions = scaler_y.inverse_transform(predictions_scaled).flatten().tolist()
        y_actual = scaler_y.inverse_transform(y_seq).flatten().tolist()
        errors = list(rmse_history)  # List RMSE per epoch
        
    elif mode == "Validasi":
        # Gunakan validation loss (simulasi)
        errors = list(rmse_history[-10:])  # 10 epoch terakhir
        predictions_scaled = model.predict(X_seq, verbose=0)
        predictions = scaler_y.inverse_transform(predictions_scaled).flatten().tolist()
        
    else:  # Testing
        # Simulasi test set dari data akhir
        test_start = int(len(X_seq) * 0.8)
        X_test_seq = X_seq[test_start:]
        y_test_seq = y_seq[test_start:]
        
        predictions_scaled = model.predict(X_test_seq, verbose=0)
        # Predictions in original scale for plotting
        predictions = scaler_y.inverse_transform(predictions_scaled).flatten().tolist()
        y_actual = scaler_y.inverse_transform(y_test_seq).flatten().tolist()

        # Compute cumulative RMSE in the SCALED space so it matches training/validation RMSE
        errors = []
        for i in range(len(predictions_scaled)):
            y_true_seg = y_test_seq[: i + 1].reshape(-1, 1)
            y_pred_seg = predictions_scaled[: i + 1].reshape(-1, 1)
            batch_rmse = float(np.sqrt(mean_squared_error(y_true_seg, y_pred_seg)))
            errors.append(batch_rmse)
        if not errors:
            test_rmse = float(np.sqrt(mean_squared_error(y_test_seq, predictions_scaled)))
            errors = [test_rmse]
    
    return predictions, errors, model_info
