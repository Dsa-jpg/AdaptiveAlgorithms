from typing import Any

import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from pandas import DataFrame
from pandas.errors import ParserError

from app.processing.normalization import zscore, minmax


def upload_data() -> DataFrame  | None:
    uploaded = st.file_uploader("Upload your time series file:", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except ParserError or ValueError:
            df = pd.read_csv(uploaded, sep=';')
        st.session_state['raw_data'] = df
        return df
    return None


def select_columns(df: DataFrame) -> tuple[Any, list, dict]:
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



def normalize_signals(df: DataFrame, y_cols: list[str]) -> tuple[DataFrame, str]:
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

@st.cache_data
def build_embedding(df: DataFrame, y_cols: list[str], lags: dict[str, int]) -> DataFrame:
    embedded = pd.DataFrame(index=df.index)

    for column in y_cols:
        for lag in range(lags[column] + 1):
            embedded[f"{column}_t-{lag}"] = df[column].shift(lag)

    embedded = embedded.dropna()
    return embedded


def plot_signals(df_plot: DataFrame, time_col: str, y_cols: list[str]) -> None:
    if y_cols:
        fig = px.line(df_plot, x=time_col, y=y_cols, title="Time Series Signals")
        st.plotly_chart(fig, width='content')
    else:
        st.info("Please select at least one signal to plot.")


def show_statistics(df: DataFrame, y_cols: list[str]) -> None:
    if y_cols:
        st.subheader("Basic Statistics")
        st.dataframe(df[y_cols].describe())

        st.subheader("Correlation Matrix")
        st.dataframe(df[y_cols].corr())

@st.cache_data
def manual_pca(x_embed: DataFrame, number_components: int)-> DataFrame:

    covX = np.corrcoef(x_embed.values.T)
    d, v = np.linalg.eig(covX)
    idx = np.argsort(d)[::-1]
    v_m = v[:, idx[:number_components]]
    X_pca = x_embed.values @ v_m

    return pd.DataFrame(X_pca, index=x_embed.index, columns=[f"PCA{i+1}" for i in range(number_components)])


def main():
    st.title("1) Load Data & Embedding")

    df = upload_data()
    if df is None:
        st.info("Upload a CSV file with time series.")
        return

    st.write("Preview of the first 10 rows:")
    st.dataframe(df.head(10))

    time_col, y_cols, lags = select_columns(df)
    df_plot, scale_mode = normalize_signals(df, y_cols)

    plot_signals(df_plot, time_col, y_cols)
    show_statistics(df_plot, y_cols)


    if y_cols:
        st.subheader("Embedding Construction")
        X_embed = build_embedding(df_plot, y_cols, lags)

        st.write(f"Embedding shape: {X_embed.shape}")
        st.dataframe(X_embed.head())

        st.session_state['X_embed'] = X_embed

        st.subheader("PCA / Dimensionality Reduction")
        use_pca = st.checkbox("Apply PCA to embedding")

        if use_pca:
            max_components = min(X_embed.shape[1], 20)
            n_components = st.slider(
                "Number of principal components",
                1, max_components, min(5, max_components)
            )

            df_pca = manual_pca(x_embed=X_embed, number_components=n_components)

            st.dataframe(df_pca.head())

            st.session_state['X_model'] = df_pca
        else:
            st.session_state['X_model'] = X_embed


        st.session_state['time_col'] = time_col
        st.session_state['y_cols'] = y_cols
        st.session_state['lags'] = lags
        st.session_state['scale_mode'] = scale_mode


if __name__ == "__main__":
    main()
