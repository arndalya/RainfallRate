# Pastikan import streamlit di paling atas
import streamlit as st
# 🔥 IMPORT MODEL TEMAN - UPDATE BACKPROP DI SINI
from models.svm_model import train as svm_train  # Teman 1
from models.bp_model import train as bp_train   # 🔥 TEMAN BACKPROP - BARIS INI BARU
from models.lstm_model import train as lstm_train  # Teman 3
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Utils functions (inline biar lengkap)
def preview_data(df):
    st.subheader("👀 Preview Data")
    st.dataframe(df.head(), use_container_width=True)
    st.metric("Total Data", len(df))
    st.metric("Features", df.shape[1]-2)  # Exclude tanggal & RR

def create_plots(errors, predictions, actual, model_name):
    col1, col2 = st.columns(2)
    
    # Grafik Error/Loss
    with col1:
        fig_loss = px.line(x=range(len(errors)), y=errors, 
                          title=f"{model_name} - Error Curve",
                          labels={'x':'Epoch', 'y':'RMSE'})
        st.plotly_chart(fig_loss, use_container_width=True)
    
    # Grafik Prediksi vs Aktual
    with col2:
        # Handle case ketika predictions lebih pendek dari actual (misal LSTM dengan sequences)
        plot_len = min(100, len(predictions), len(actual))
        
        fig_pred = px.scatter(x=actual[:plot_len], y=predictions[:plot_len],
                            title=f"{model_name} - Prediksi vs Aktual",
                            labels={'x':'Aktual', 'y':'Prediksi'})
        fig_pred.add_shape(type="line", x0=actual[:plot_len].min(), y0=actual[:plot_len].min(),
                          x1=actual[:plot_len].max(), y1=actual[:plot_len].max(), line=dict(color="red"))
        st.plotly_chart(fig_pred, use_container_width=True)

st.set_page_config(page_title="Rainfall Rate Predictor", layout="wide", page_icon="🌧️")
st.title("🌧️ Rainfall Rate Predictor")
st.markdown("---")

# ========================================================
# SIDEBAR NAVIGASI
# ========================================================
st.sidebar.header("⚙️ Kontrol Panel")
mode = st.sidebar.selectbox("Pilih Mode", ["Training", "Validasi", "Testing"])
model_choice = st.sidebar.selectbox("Pilih Model", ["SVM/SVR", "Backpropagation", "LSTM"])
lr = st.sidebar.number_input("Learning Rate", 0.001, 0.1, 0.01, 0.001)
steps_epochs = st.sidebar.number_input("Epochs", 50, 1000, 100)



# 🔥 FITUR GABUNGAN - BUTTON PERBANDINGAN
if st.sidebar.button("📊 Bandingkan Semua Model", use_container_width=True):
    st.session_state.compare_mode = True
else:
    st.session_state.compare_mode = False

# ========================================================
# UPLOAD & PREVIEW DATA
# ========================================================



uploaded_file = st.file_uploader("📁 Upload CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=';', decimal=',')  # CSV BMKG menggunakan semicolon & comma decimal
    st.session_state.df = df
    
    col1, col2 = st.columns([2, 1])
    with col1: preview_data(df)
    with col2: 
        st.info("**Target:** Rata-rata Curah Hujan (RR)")
        st.info("**Features:** Tx, Tn, RH_avg, ss")

    # Prepare features & target
    feature_cols = ['TX', 'TN', 'RH_AVG', 'SS']  # Sesuai dengan kolom CSV BMKG
    target_col = "RR" 
    X = df[feature_cols].fillna(0)
    y = df['RR'].fillna(0)
    
    # ========================================================
    # 🔥 MODE PERBANDINGAN 3 MODEL - UPDATE BACKPROP DI SINI
    # ========================================================
    if st.session_state.get('compare_mode', False):
        st.header("📈 Perbandingan 3 Model ML")
        
        models_data = {}
        with st.spinner("🔄 Menjalankan SVM + Backpropagation + LSTM..."):
            # 🔥 BACKPROP UPDATED - BARIS INI BARU
            svm_pred, svm_err, _ = svm_train(X, y, lr, steps_epochs, mode)
            bp_pred, bp_err, _ = bp_train(X, y, lr, steps_epochs, mode)   # 🔥 UPDATE
            lstm_pred, lstm_err, _ = lstm_train(X, y, lr, steps_epochs, mode)
            
            models_data = {
                'SVM/SVR': {'pred': svm_pred, 'error': svm_err[-1]},
                'Backpropagation': {'pred': bp_pred, 'error': bp_err[-1]},  # 🔥 UPDATE
                'LSTM': {'pred': lstm_pred, 'error': lstm_err[-1]}
            }
        
        # Grafik 1: Bar Chart Akurasi
        col1, col2 = st.columns(2)
        with col1:
            accuracies = {k: max(0, (1-v['error'])*100) for k,v in models_data.items()}
            fig_bar = px.bar(x=list(accuracies.keys()), y=list(accuracies.values()),
                           title="🏆 Akurasi Perbandingan Model (%)",
                           color=list(accuracies.values()),
                           color_continuous_scale='Viridis')
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Grafik 2: Prediksi vs Aktual
        with col2:
            fig_line = go.Figure()
            for model_name, data in models_data.items():
                fig_line.add_trace(go.Scatter(x=y[:100], y=data['pred'][:100],
                                            mode='lines+markers', name=model_name,
                                            line=dict(width=2)))
            fig_line.add_trace(go.Scatter(x=y[:100], y=y[:100], mode='lines',
                                        name='Aktual', line=dict(dash='dash', color='black')))
            fig_line.update_layout(title="📊 Prediksi vs Aktual (100 sampel)")
            st.plotly_chart(fig_line, use_container_width=True)
        
        # Tabel Perbandingan
        st.subheader("📋 Detail Performa Model")
        comparison_df = pd.DataFrame({
            'Model': list(models_data.keys()),
            'Error Akhir (RMSE)': [f"{v['error']:.4f}" for v in models_data.values()],
            'Akurasi (%)': [f"{accuracies[k]:.2f}" for k in accuracies.keys()]
        })
        st.dataframe(comparison_df.style.highlight_min(subset=['Error Akhir (RMSE)'])
                     .highlight_max(subset=['Akurasi (%)']), use_container_width=True)
        
        # Download gabungan
        # Pastikan semua data punya panjang sama
        min_len = min(len(y), 
                     len(models_data['SVM/SVR']['pred']),
                     len(models_data['Backpropagation']['pred']),
                     len(models_data['LSTM']['pred']))
        
        all_results = pd.DataFrame({
            'Aktual': y.values[:min_len],
            'SVM_Pred': models_data['SVM/SVR']['pred'][:min_len],
            'BP_Pred': models_data['Backpropagation']['pred'][:min_len],
            'LSTM_Pred': models_data['LSTM']['pred'][:min_len]
        })
        st.download_button("💾 Download Semua Hasil", 
                          all_results.to_csv(index=False),
                          "perbandingan_3_model.csv", use_container_width=True)
    
    # ========================================================
    # MODE SINGLE MODEL - UPDATE BACKPROP DI SINI
    # ========================================================
    else:
        if st.button(f"🚀 Mulai {mode} - {model_choice}", type="primary", use_container_width=True):
            with st.spinner(f"Training {model_choice}..."):
                # 🔥 BACKPROP UPDATED - BARIS INI BARU
                if model_choice == "SVM/SVR":
                    results, errors, model_info = svm_train(X, y, lr, steps_epochs, mode)
                elif model_choice == "Backpropagation":
                    results, errors, model_info = bp_train(X, y, lr, steps_epochs, mode)  # 🔥 UPDATE
                else:  # LSTM
                    results, errors, model_info = lstm_train(X, y, lr, steps_epochs, mode)
                
                # Simpan untuk testing berulang
                st.session_state.model_info = model_info
                st.session_state.results = results
                st.session_state.errors = errors
            
            # Tampilkan hasil
            st.success("✅ Selesai!")
            
            # Pastikan hasil & target punya panjang sama sebelum metrics/plot
            results_arr = np.array(results).flatten()
            y_arr = np.array(y).flatten()
            min_len_plot = min(len(results_arr), len(y_arr))
            if min_len_plot == 0:
                st.error("❌ Tidak ada data untuk ditampilkan.")
                raise st.stop()
            results_arr = results_arr[:min_len_plot]
            y_arr = y_arr[:min_len_plot]

            # Hitung metrics lengkap
            error_awal = errors[0] if len(errors) > 0 else 0
            error_akhir = errors[-1] if len(errors) > 0 else 0
            error_rata_rata = np.mean(errors) if len(errors) > 0 else 0

            # Tampilkan metrics dalam 4 kolom (kembalikan akurasi berbasis RMSE seperti semula)
            akurasi = max(0, (1 - error_akhir) * 100)
            # st.metric("RMSE Akhir", f"{error_akhir:.4f}")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Error Awal", f"{error_awal:.4f}")
            with col2:
                st.metric("📊 Error Akhir", f"{error_akhir:.4f}")
            with col3:
                st.metric("📊 Error Rata-rata", f"{error_rata_rata:.4f}")
            with col4:
                st.metric("🎯 Akurasi", f"{akurasi:.1f}%")

            # Grafik hasil (gunakan arrays yang sudah disesuaikan panjangnya)
            create_plots(errors, results_arr, y_arr, model_choice)
            
            # Download single model
            # Pastikan hasil dan target punya panjang sama
            if len(results) != len(y):
                st.warning(f"⚠️ Mismatch data: hasil ({len(results)}) vs target ({len(y)}). Menggunakan data yang lebih pendek.")
                min_len = min(len(results), len(y))
                results = results[:min_len]
                y = y[:min_len]
            
            csv_results = pd.DataFrame({'Prediksi': results, 'Aktual': y})
            st.download_button("💾 Download Hasil CSV", 
                             csv_results.to_csv(index=False),
                             f"hasil_{model_choice}_{mode.lower()}.csv",
                             use_container_width=True)

else:
    st.info("👆 Silakan upload file CSV terlebih dahulu")
    st.markdown("**Format kolom yang dibutuhkan:** Tx, Tn, RH_avg, ss, RR")

# def train():
#     # Fungsi dummy, nanti diganti dengan kode asli
#     pass
