import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train(X, y, mode="Training", C=10, gamma=0.05, epsilon=0.1):
    """
    SVR Regression (non-epoch model)
    """

    # ===============================
    # SPLIT DATA (TIME-SERIES STYLE)
    # ===============================
    split = int(len(X) * 0.8)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # ===============================
    # SCALING (ROBUST → tahan outlier hujan)
    # ===============================
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled  = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(
        y_train.values.reshape(-1, 1)
    ).ravel()

    # ===============================
    # TRAIN SVR
    # ===============================
    model = SVR(
        kernel="rbf",
        C=C,
        gamma=gamma,
        epsilon=epsilon
    )
    model.fit(X_train_scaled, y_train_scaled)

    # ===============================
    # PILIH MODE
    # ===============================
    if mode == "Training":
        X_use = X_train_scaled
        y_use = y_train
    else:
        X_use = X_test_scaled
        y_use = y_test

    # ===============================
    # PREDIKSI
    # ===============================
    pred_scaled = model.predict(X_use)
    preds = scaler_y.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).flatten()

    # ===============================
    # METRICS REGRESI
    # ===============================
    rmse = np.sqrt(mean_squared_error(y_use, preds))
    mae  = mean_absolute_error(y_use, preds)
    r2   = r2_score(y_use, preds)

    model_info = {
        "hyperparameter": {
            "C": C,
            "gamma": gamma,
            "epsilon": epsilon
        },
        "metrics": {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }
    }

    # SVR tidak punya error curve → return None
    return preds.tolist(), None, model_info
