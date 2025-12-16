import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def train(X, y, lr, epochs, mode="Training"):
    """
    Backpropagation (MLP)
    Urutan dan output DISESUAIKAN dengan LSTM
    """

    # 1. Mapping kolom (SAMA)
    feature_map = {'Tx': 'TX', 'Tn': 'TN', 'RH_avg': 'RH_AVG', 'ss': 'SS'}
    X_renamed = X.rename(columns=feature_map)

    # 2. Normalisasi data (SAMA)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_renamed)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    # 3. Bangun model BACKPROPAGATION (MLP)
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(X_scaled.shape[1],)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))  # Output RR

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='mse'
    )

    # 4. Training (SAMA)
    early_stop = EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_scaled,
        y_scaled,
        epochs=epochs,
        batch_size=32,
        verbose=0,
        callbacks=[early_stop]
    )

    # 5. Simpan info model (SAMA FORMAT)
    model_info = {
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'history': history.history
    }

    # 6. Hitung RMSE per epoch
    loss_history = np.array(history.history.get('loss', []))
    rmse_history = np.sqrt(loss_history).tolist()

    # ================= MODE HANDLING =================
    if mode == "Training":
        predictions_scaled = model.predict(X_scaled, verbose=0)
        predictions = scaler_y.inverse_transform(predictions_scaled).flatten().tolist()
        errors = rmse_history

    elif mode == "Validasi":
        predictions_scaled = model.predict(X_scaled, verbose=0)
        predictions = scaler_y.inverse_transform(predictions_scaled).flatten().tolist()
        errors = rmse_history[-10:]

    else:  # Testing
        test_start = int(len(X_scaled) * 0.8)
        X_test = X_scaled[test_start:]
        y_test = y_scaled[test_start:]

        predictions_scaled = model.predict(X_test, verbose=0)
        predictions = scaler_y.inverse_transform(predictions_scaled).flatten().tolist()

        errors = []
        for i in range(len(predictions_scaled)):
            rmse = np.sqrt(
                mean_squared_error(
                    y_test[:i+1],
                    predictions_scaled[:i+1]
                )
            )
            errors.append(float(rmse))

    return predictions, errors, model_info
