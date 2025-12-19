import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train(X, y, mode="Training", C=10, gamma=0.05, epsilon=0.1):

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler_X = RobustScaler()
    scaler_y = RobustScaler()

    Xtr = scaler_X.fit_transform(X_train)
    Xte = scaler_X.transform(X_test)
    ytr = scaler_y.fit_transform(y_train.values.reshape(-1,1)).ravel()

    model = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)
    model.fit(Xtr, ytr)

    X_use = Xtr if mode == "Training" else Xte
    y_use = y_train if mode == "Training" else y_test

    pred_scaled = model.predict(X_use)
    preds = scaler_y.inverse_transform(
        pred_scaled.reshape(-1,1)
    ).flatten()

    info = {
        "hyperparameter": {
            "C": C,
            "gamma": gamma,
            "epsilon": epsilon
        },
        "metrics": {
            "RMSE": np.sqrt(mean_squared_error(y_use, preds)),
            "MAE": mean_absolute_error(y_use, preds),
            "R2": r2_score(y_use, preds)
        }
    }

    return preds.tolist(), None, info
