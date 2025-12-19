# =========================================
# app.py
# =========================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from models.lstm_model import train as lstm_train
from models.bp_model import train as bp_train
from models.svr_model import train as svr_train

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Rainfall Rate Predictor",
    page_icon="🌧️",
    layout="wide"
)

st.title("🌧️ Rainfall Rate Predictor")
st.markdown("---")

# =========================================
# SIDEBAR
# =========================================
st.sidebar.header("⚙️ Kontrol Panel")

mode = st.sidebar.selectbox(
    "Mode",
    ["Training", "Validasi", "Testing"]
)

model_choice = st.sidebar.selectbox(
    "Model",
    ["LSTM", "Backpropagation", "SVR"]
)

# ===== Parameter Dinamis =====
if model_choice in ["LSTM", "Backpropagation"]:
    lr = st.sidebar.selectbox("Learning Rate", [0.001, 0.005, 0.01])
    epochs = st.sidebar.selectbox("Epoch", [50, 100, 200])
else:
    C = st.sidebar.selectbox("C", [1, 5, 10, 20])
    gamma = st.sidebar.selectbox("Gamma", [0.01, 0.05, 0.1])
    epsilon = st.sidebar.selectbox("Epsilon", [0.01, 0.1, 0.2])

compare_mode = st.sidebar.button("📊 Bandingkan Semua Model")

# ===============================
# UPLOAD DATA
# ===============================
uploaded_file = st.file_uploader("📁 Upload CSV BMKG", type="csv")

if not uploaded_file:
    st.info("👆 Silakan upload file CSV terlebih dahulu")
    st.markdown("""
    **Keterangan:**
    - TN (Temperatur Minimum)
    - TX (Temperatur Maksimum)
    - TAVG (Temperatur Rata-rata)
    - RH_AVG (Kelembapan Rata-rata)
    - SS (Lama Penyinaran Matahari)
    - RR (Curah Hujan)
    """)
    st.stop()
    
df = pd.read_csv(uploaded_file, sep=";", decimal=",")

# =========================================
# PREVIEW DATA
# =========================================
st.subheader("👀 Preview Data")
st.dataframe(df.head(), use_container_width=True)

c1, c2 = st.columns(2)
c1.metric("Total Data", len(df))
c2.metric("Jumlah Feature", 5)

st.info("🎯 Target: RR (Curah Hujan)")
st.info("📌 Features: TN, TX, TAVG, RH_AVG, SS")

# =========================================
# PREPARE DATA
# =========================================
features = ["TN", "TX", "TAVG", "RH_AVG", "SS"]
X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
y = pd.to_numeric(df["RR"], errors="coerce").fillna(0)

# =========================================
# MODE: BANDINGKAN SEMUA MODEL
# =========================================
if compare_mode:
    st.header("📊 Perbandingan Prediksi Semua Model")

    with st.spinner("Menjalankan LSTM, Backpropagation, dan SVR..."):
        pred_lstm, _, _ = lstm_train(X, y, 0.001, 50, mode)
        pred_bp, _, _ = bp_train(X, y, 0.001, 50, mode)
        pred_svr, _, _ = svr_train(X, y, mode)

    # Samakan panjang data
    min_len = min(len(pred_lstm), len(pred_bp), len(pred_svr))
    idx = list(range(min_len))

    fig_compare = go.Figure()

    fig_compare.add_trace(go.Scatter(
        x=idx,
        y=pred_lstm[:min_len],
        mode="lines",
        name="LSTM",
        line=dict(color="dodgerblue")
    ))

    fig_compare.add_trace(go.Scatter(
        x=idx,
        y=pred_bp[:min_len],
        mode="lines",
        name="Backpropagation",
        line=dict(color="#FF1654")
    ))

    fig_compare.add_trace(go.Scatter(
        x=idx,
        y=pred_svr[:min_len],
        mode="lines",
        name="SVR",
        line=dict(color="orange")
    ))

    fig_compare.update_layout(
        title="Perbandingan Prediksi Curah Hujan (RR)",
        xaxis_title="Index Data",
        yaxis_title="RR (Prediksi)",
        legend_title="Model"
    )

    st.plotly_chart(fig_compare, use_container_width=True)

    # Download hasil perbandingan
    compare_df = pd.DataFrame({
        "LSTM": pred_lstm[:min_len],
        "Backpropagation": pred_bp[:min_len],
        "SVR": pred_svr[:min_len]
    })

    st.download_button(
        "💾 Download CSV Perbandingan",
        compare_df.to_csv(index=False),
        file_name="perbandingan_model.csv",
        use_container_width=True
    )

    st.stop()

# =========================================
# MODE: SINGLE MODEL
# =========================================
if st.button(f"🚀 Jalankan {model_choice}", use_container_width=True):

    with st.spinner(f"Menjalankan {model_choice}..."):

        if model_choice == "LSTM":
            preds, errors, _ = lstm_train(X, y, lr, epochs, mode)

        elif model_choice == "Backpropagation":
            preds, errors, _ = bp_train(X, y, lr, epochs, mode)

        else:
            preds, _, model_info = svr_train(
                X, y, mode, C=C, gamma=gamma, epsilon=epsilon
            )

    st.success("✅ Proses selesai")

    preds = np.array(preds)
    y_plot = y.values[:len(preds)]

    # ================= METRICS =================
    if model_choice != "SVR":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RMSE Awal", f"{errors[0]:.4f}")
        c2.metric("RMSE Akhir", f"{errors[-1]:.4f}")
        c3.metric("RMSE Rata-rata", f"{np.mean(errors):.4f}")
        c4.metric("Akurasi (%)", f"{max(0,(1-errors[-1])*100):.1f}%")
    else:
        m = model_info["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{m['RMSE']:.4f}")
        c2.metric("MAE", f"{m['MAE']:.4f}")
        c3.metric("R²", f"{m['R2']:.4f}")

    # ================= VISUALISASI =================
    st.markdown("## 📊 Visualisasi Prediksi")

    col1, col2 = st.columns(2)

    with col1:
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=y_plot,
            y=preds,
            mode="markers",
            name="Prediksi",
            marker=dict(color="dodgerblue")
        ))
        fig_scatter.add_trace(go.Scatter(
            x=y_plot,
            y=y_plot,
            mode="lines",
            name="Aktual (Ideal)",
            line=dict(color="yellow", dash="dash")
        ))
        fig_scatter.update_layout(
            title="Scatter Prediksi vs Aktual",
            xaxis_title="Aktual (RR)",
            yaxis_title="Prediksi (RR)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            y=y_plot,
            mode="lines",
            name="Aktual",
            line=dict(color="grey")
        ))
        fig_line.add_trace(go.Scatter(
            y=preds,
            mode="lines",
            name="Prediksi",
            line=dict(color="dodgerblue")
        ))
        fig_line.update_layout(
            title="Line Chart Prediksi vs Aktual",
            xaxis_title="Index Data",
            yaxis_title="RR"
        )
        st.plotly_chart(fig_line, use_container_width=True)

    if model_choice != "SVR":
        fig_err = px.line(
            x=list(range(1, len(errors)+1)),
            y=errors,
            labels={
                "x": "Epoch / Iterasi",
                "y": "RMSE"
            },
            title=f"{model_choice} – Error Curve (RMSE)"
        )
        st.plotly_chart(fig_err, use_container_width=True)

    # ================= DOWNLOAD =================
    out_df = pd.DataFrame({
        "Aktual": y_plot,
        "Prediksi": preds
    })

    st.download_button(
        "💾 Download CSV Hasil",
        out_df.to_csv(index=False),
        file_name=f"hasil_{model_choice.lower()}.csv",
        use_container_width=True
    )
