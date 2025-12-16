# Teman 1 isi implementasi SVM/SVR
def train(X, y, lr, epochs, mode):
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if mode == "Training":
        model = SVR(kernel='rbf', C=1.0)
        model.fit(X_scaled, y)
        pred = model.predict(X_scaled)
        errors = [mean_squared_error(y, pred)]  # Single error untuk SVM
        return pred, errors, model
    # Tambah logic validasi/testing serupa
