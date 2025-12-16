# utils.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def preview_data(df: pd.DataFrame):
    st.subheader("👀 Preview Data")
    st.dataframe(df.head(), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Data", len(df))
    with col2:
        st.metric("Jumlah Kolom", df.shape[1])

def create_plots(errors, predictions, actual, model_name: str):
    errors = np.array(errors, dtype=float)
    predictions = np.array(predictions).flatten()
    actual = np.array(actual).flatten()

    col1, col2 = st.columns(2)

    # Grafik error
    with col1:
        fig_loss = px.line(
            x=list(range(1, len(errors) + 1)),
            y=errors,
            labels={"x": "Epoch/Step", "y": "Error (RMSE / Loss)"},
            title=f"{model_name} - Kurva Error",
        )
        st.plot
