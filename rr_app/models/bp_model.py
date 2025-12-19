import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def train(X, y, lr, epochs, mode="Training"):
    """
    Backpropagation (MLP)
    FIXED: Tidak training ulang saat Validasi / Testing
    """

    # ===============================
    # NORMALISASI
    # ===============================
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    # ===============================
    # BUILD MODEL
    # ===============================
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(X_scaled.shape[1],)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='mse'
    )

    # ===============================
    # TRAINING (HANYA SEKALI)
    # ===============================
    history = None
    rmse_history = []

    if mode == "Training":
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

        loss_history = np.array(history.history['loss'])
        rmse_history = np.sqrt(loss_history).tolist()

    # ===============================
    # PREDIKSI
    # ===============================
    if mode == "Testing":
        split = int(len(X_scaled) * 0.8)
        X_use = X_scaled[split:]
        y_use = y_scaled[split:]
    else:
        X_use = X_scaled
        y_use = y_scaled

    preds_scaled = model.predict(X_use, verbose=0)
    preds = scaler_y.inverse_transform(preds_scaled).flatten()

    # ===============================
    # ERROR HANDLING
    # ===============================
    if mode != "Training":
        # Hitung RMSE kumulatif (simulasi error curve)
        rmse_history = []
        for i in range(len(preds_scaled)):
            rmse = np.sqrt(
                mean_squared_error(
                    y_use[:i+1],
                    preds_scaled[:i+1]
                )
            )
            rmse_history.append(float(rmse))

    model_info = {
        "model": model,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y
    }

    return preds.tolist(), rmse_history, model_info
