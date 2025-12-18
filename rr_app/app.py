# ===============================
# app.py (FINAL & FIXED)
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from models.svr_model import train as svr_train
from models.bp_model import train as bp_train
from models.lstm_model import train as lstm_train

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Rainfall Rate Predictor",
    page_icon="🌧️",
    layout="wide"
)

st.title("🌧️ Rainfall Rate Predictor")
st.markdown("---")

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("⚙️ Kontrol Panel")

mode = st.sidebar.selectbox(
    "Pilih Mode",
    ["Training", "Validasi", "Testing"]
)

model_choice = st.sidebar.selectbox(
    "Pilih Model",
    ["SVR", "Backpropagation", "LSTM"]
)

lr = st.sidebar.number_input(
    "Learning Rate (NN only)",
    min_value=0.001,
    max_value=0.1,
    value=0.01,
    step=0.001
)

epochs = st.sidebar.number_input(
    "Epochs (NN only)",
    min_value=10,
    max_value=1000,
    value=100
)

compare_mode = st.sidebar.button(
    "📊 Bandingkan Semua Model",
    use_container_width=True
)

st.sidebar.markdown("---")

# ===============================
# UPLOAD FILE (INI WAJIB BENAR)
# ===============================
uploaded_file = st.file_uploader(
    "📁 Upload CSV BMKG",
    type="csv"
)

if uploaded_file is None:
    st.info("👆 Silakan upload file CSV terlebih dahulu")
    st.stop()

# ===============================
# LOAD DATA (FIX NAME ERROR)
# ===============================
df = pd.read_csv(uploaded_file, sep=";", decimal=",")

# ===============================
# PREVIEW DATA
# ===============================
st.subheader("👀 Preview Data")
st.dataframe(df.head(), use_container_width=True)

c1, c2 = st.columns(2)
c1.metric("Total Data", len(df))
c2.metric("Jumlah Feature", 5)

st.markdown("### 🎯 Informasi Data")
st.info("**Target:** RR (Rainfall Rate / Curah Hujan)")
st.info("**Features:** TN, TX, TAVG, RH_AVG, SS")

# ===============================
# PREPARE DATA
# ===============================
feature_cols = ["TN", "TX", "TAVG", "RH_AVG", "SS"]
X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
y = pd.to_numeric(df["RR"], errors="coerce").fillna(0)

st.markdown("---")

# ===============================
# MODE PERBANDINGAN
# ===============================
if compare_mode:
    st.header("📈 Perbandingan 3 Model")

    with st.spinner("Menjalankan semua model..."):
        svr_pred, _, svr_info = svr_train(X, y, mode)
        bp_pred, bp_err, _ = bp_train(X, y, lr, epochs, mode)
        lstm_pred, lstm_err, _ = lstm_train(X, y, lr, epochs, mode)

    rmse = {
        "SVR": svr_info["metrics"]["RMSE"],
        "Backpropagation": bp_err[-1],
        "LSTM": lstm_err[-1]
    }

    fig_bar = px.bar(
        x=list(rmse.keys()),
        y=list(rmse.values()),
        title="Perbandingan RMSE"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    min_len = min(len(y), len(svr_pred), len(bp_pred), len(lstm_pred))
    out_df = pd.DataFrame({
        "Aktual": y[:min_len],
        "SVR": svr_pred[:min_len],
        "Backpropagation": bp_pred[:min_len],
        "LSTM": lstm_pred[:min_len]
    })

    st.download_button(
        "💾 Download Perbandingan CSV",
        out_df.to_csv(index=False),
        "perbandingan_model.csv",
        use_container_width=True
    )

# ===============================
# MODE SINGLE MODEL
# ===============================
else:
    if st.button(
        f"🚀 Jalankan {model_choice}",
        type="primary",
        use_container_width=True
    ):
        with st.spinner(f"Menjalankan {model_choice}..."):

            if model_choice == "SVR":
                preds, _, model_info = svr_train(X, y, mode)

            elif model_choice == "Backpropagation":
                preds, errors, _ = bp_train(X, y, lr, epochs, mode)

            else:
                preds, errors, _ = lstm_train(X, y, lr, epochs, mode)

        st.success("✅ Proses selesai")

        preds = np.array(preds)
        y_plot = y.values[:len(preds)]

        # ===== METRICS =====
        if model_choice == "SVR":
            m = model_info["metrics"]
            h = model_info["hyperparameter"]

            st.subheader("🔧 Hyperparameter SVR")
            st.json(h)

            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE", f"{m['RMSE']:.4f}")
            c2.metric("MAE", f"{m['MAE']:.4f}")
            c3.metric("R²", f"{m['R2']:.4f}")

            st.info(
                "ℹ️ R² negatif berarti SVR lebih buruk dari prediksi rata-rata. "
                "Ini normal pada data curah hujan."
            )

        else:
            error_awal = errors[0]
            error_akhir = errors[-1]
            error_rata = np.mean(errors)
            akurasi = max(0, (1 - error_akhir) * 100)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Error Awal (RMSE)", f"{error_awal:.4f}")
            c2.metric("Error Akhir (RMSE)", f"{error_akhir:.4f}")
            c3.metric("Error Rata-rata", f"{error_rata:.4f}")
            c4.metric("Akurasi (%)", f"{akurasi:.1f}%")

        # ===== PLOT =====
        fig = px.scatter(
            x=y_plot,
            y=preds,
            labels={"x": "Aktual", "y": "Prediksi"},
            title=f"{model_choice} – Prediksi vs Aktual"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ===== DOWNLOAD =====
        out_df = pd.DataFrame({
            "Aktual": y_plot,
            "Prediksi": preds
        })

        st.download_button(
            "💾 Download Hasil CSV",
            out_df.to_csv(index=False),
            f"hasil_{model_choice.lower()}.csv",
            use_container_width=True
        )
