from typing import Any

import streamlit as st
import pandas as pd
import plotly.express as px
from pandas import DataFrame

from app.processing.normalization import zscore, minmax


def upload_data() -> DataFrame  | None:
    uploaded = st.file_uploader("Upload your time series file:", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            df = pd.read_csv(uploaded, sep=';')
        st.session_state['raw_data'] = df
        return df
    return None


def select_columns(df: DataFrame) -> tuple[Any, list, dict] | None:
    st.subheader("Select Time Column (X)")
    time_col = st.selectbox("Choose the time column:", df.columns)

    st.subheader("Select Signals (Y)")
    y_cols = st.multiselect(
        "Choose signals for embedding and plotting:",
        [c for c in df.columns if c != time_col]
    )

    lags = {}
    if y_cols:
        st.subheader("Select Lags for Each Signal")
        for col in y_cols:
            lag = st.slider(f"Lags for {col}:", 1, 20, 1)
            lags[col] = lag
    return time_col, y_cols, lags


def normalize_signals(df, y_cols):
    st.subheader("Optional Signal Normalization")
    scale_mode = st.radio(
        "Choose normalization method:",
        ["None", "Z-score", "Min-Max"]
    )

    df_plot = df.copy()
    if y_cols:
        if scale_mode == "Z-score":
            for col in y_cols:
                df_plot[col] = zscore(df[col])
        elif scale_mode == "Min-Max":
            for col in y_cols:
                df_plot[col] = minmax(df[col])
    return df_plot, scale_mode

def plot_signals(df_plot, time_col, y_cols) -> None:
    if y_cols:
        fig = px.line(df_plot, x=time_col, y=y_cols, title="Time Series Signals")
        st.plotly_chart(fig, width='content')
    else:
        st.info("Please select at least one signal to plot.")


def show_statistics(df, y_cols):
    if y_cols:
        st.subheader("Basic Statistics")
        st.dataframe(df[y_cols].describe())
        st.subheader("Correlation Matrix")
        corr = df[y_cols].corr()
        st.dataframe(corr)

def main():
    st.title("1) Load Data")

    df = upload_data()
    if df is not None:
        st.write("Preview of the first 10 rows:")
        st.dataframe(df.head(10))

        time_col, y_cols, lags = select_columns(df)
        df_plot, scale_mode = normalize_signals(df, y_cols)
        plot_signals(df_plot, time_col, y_cols)
        show_statistics(df_plot, y_cols)

        st.session_state['time_col'] = time_col
        st.session_state['y_cols'] = y_cols
        st.session_state['lags'] = lags
    else:
        st.info("Upload a CSV or TXT file with time series.")


if __name__ == "__main__":
    main()
