import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split


def train(X, y):
    """
    SVM - KLASIFIKASI CURAH HUJAN
    Target asli RR (kontinu) → dikonversi ke kelas hujan
    """

    # ===============================
    # 1. KONVERSI RR → KELAS HUJAN
    # ===============================
    bins = [-1, 5, 20, 50, 100, float("inf")]
    class_names = [
        "Sangat Ringan",
        "Ringan",
        "Sedang",
        "Lebat",
        "Sangat Lebat"
    ]

    y_cat = pd.cut(y, bins=bins, labels=class_names)

    label_map = {name: i for i, name in enumerate(class_names)}
    y_label = y_cat.map(label_map)

    # Buang data gagal label
    valid_idx = y_label.notna()
    X = X.loc[valid_idx]
    y_label = y_label.loc[valid_idx].astype(int)

    # ===============================
    # 2. SPLIT DATA (TIME SERIES SAFE)
    # ===============================
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_label, test_size=0.2, shuffle=False
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, shuffle=False
    )

    # ===============================
    # 3. NORMALISASI
    # ===============================
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # ===============================
    # 4. MODEL SVM
    # ===============================
    model = SVC(
        kernel="rbf",
        C=100,
        gamma=1,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # ===============================
    # 5. PREDIKSI TEST
    # ===============================
    y_pred = model.predict(X_test)

    # ===============================
    # 6. METRICS KLASIFIKASI
    # ===============================
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred) * 100,
        "precision": precision_score(y_test, y_pred, average="macro") * 100,
        "recall": recall_score(y_test, y_pred, average="macro") * 100,
        "f1_score": f1_score(y_test, y_pred, average="macro") * 100,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "class_names": class_names,
        "classification_report": classification_report(
            y_test, y_pred, target_names=class_names
        )
    }

    return y_pred.tolist(), metrics
