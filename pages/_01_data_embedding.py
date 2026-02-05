import streamlit as st
from app.processing.io import load_csv
from app.processing.normalization import zscore, minmax
from app.processing.embedding import build_lag_embedding
from app.processing.pca import pca_manual


def upload_data():
    uploaded = st.file_uploader("Upload your time series file:", type=["csv"])
    if uploaded:
        df = load_csv(uploaded)
        st.session_state['raw_data'] = df
        return df
    return None


def select_columns(df):
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


def plot_signals(df_plot, time_col, y_cols):
    if y_cols:
        import plotly.express as px
        fig = px.line(df_plot, x=time_col, y=y_cols, title="Time Series Signals")
        st.plotly_chart(fig, width='content')
    else:
        st.info("Please select at least one signal to plot.")


def show_statistics(df, y_cols):
    if y_cols:
        st.subheader("Basic Statistics")
        st.dataframe(df[y_cols].describe())

        st.subheader("Correlation Matrix")
        st.dataframe(df[y_cols].corr())


def main():
    st.title("1) Load Data & Embedding")

    st.markdown("""
    ### Purpose of this step
    This page prepares input data for adaptive modeling:
    - uploading time series data
    - selecting signals and lags
    - optional normalization
    - constructing lag embeddings
    - optional PCA for dimensionality reduction

    The resulting embedding matrix is used as input for adaptive models in the next steps.
    """)

    # Upload CSV
    df = upload_data()
    if df is None:
        st.info("Upload a CSV file with time series to continue.")
        return

    st.write("Preview of the first 10 rows:")
    st.dataframe(df.head(10))

    # Column selection
    time_col, y_cols, lags = select_columns(df)

    # Normalize signals
    df_plot, scale_mode = normalize_signals(df, y_cols)

    # Plot selected signals
    plot_signals(df_plot, time_col, y_cols)
    show_statistics(df_plot, y_cols)

    # Build lag embedding
    if y_cols:
        st.subheader("Embedding Construction")
        X_embed = build_lag_embedding(df_plot, y_cols, lags)
        st.write(f"Embedding shape: {X_embed.shape}")
        st.dataframe(X_embed.head())
        st.session_state['X_embed'] = X_embed

        # PCA / Dimensionality reduction
        st.subheader("PCA / Dimensionality Reduction")
        use_pca = st.checkbox("Apply PCA to embedding")

        if use_pca:
            total_components = X_embed.shape[1]
            X_full_pca = pca_manual(X_embed, total_components)

            options = [f"PCA{i + 1}" for i in range(total_components)]
            selected_components = st.multiselect(
                "Select PCA components to use:",
                options,
                default=options[:5]
            )

            if selected_components:
                X_pca_selected = X_full_pca[selected_components]
                st.dataframe(X_pca_selected.head())
                st.session_state['X_model'] = X_pca_selected
            else:
                st.warning("No PCA components selected, using all components.")
                st.session_state['X_model'] = X_full_pca

        # Save selection to session_state
        st.session_state['time_col'] = time_col
        st.session_state['y_cols'] = y_cols
        st.session_state['lags'] = lags
        st.session_state['scale_mode'] = scale_mode


if __name__ == "__main__":
    main()
