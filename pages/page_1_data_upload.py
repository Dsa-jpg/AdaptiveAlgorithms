from typing import Any

import streamlit as st
import pandas as pd
import plotly.express as px
from pandas import DataFrame

from app.processing.normalization import zscore, minmax


def upload_data() -> DataFrame  | None:
    uploaded = st.file_uploader("Upload your time series file:", type=["csv", "txt"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            df = pd.read_csv(uploaded, sep=';')
        st.session_state['raw_data'] = df
        return df
    return None


def select_columns(df: DataFrame)-> tuple[Any,None] | None:
    st.subheader("Select Time Column (X)")
    time_col = st.selectbox("Choose the time column:", df.columns)

    st.subheader("Select Signals (Y)")
    y_cols = st.multiselect(
        "Choose signals to plot:",
        [c for c in df.columns if c != time_col]
    )
    return time_col, y_cols


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
        fig = px.line(df_plot, x=time_col, y=y_cols)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select at least one signal to plot.")


def main():
    st.title("1) Load Data")

    df = upload_data()
    if df is not None:
        st.write("Preview of the first 10 rows:")
        st.dataframe(df.head(10))

        time_col, y_cols = select_columns(df)
        df_plot, scale_mode = normalize_signals(df, y_cols)
        plot_signals(df_plot, time_col, y_cols)
    else:
        st.info("Upload a CSV or TXT file with time series.")


if __name__ == "__main__":
    main()
