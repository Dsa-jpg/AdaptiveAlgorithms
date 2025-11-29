import streamlit as st
import pandas as pd
import plotly.express as px

st.title("1) Nahrání dat")

uploaded = st.file_uploader("Nahraj CSV s časovými řadami", type=["csv", "txt"])

if uploaded:
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        df = pd.read_csv(uploaded, sep=';')

    st.session_state['raw_data'] = df

    st.write("Náhled dat (prvních 10 řádků):")
    st.dataframe(df.head(10))

    st.subheader("2) Výběr časové osy (X)")
    time_col = st.selectbox("Vyber sloupec s časem", df.columns)

    st.subheader("3) Výběr signálů (Y)")
    y_cols = st.multiselect(
        "Vyber signály k vykreslení",
        [c for c in df.columns if c != time_col]
    )

    st.subheader("4) Normalizace signálů (volitelné)")
    scale_mode = st.radio(
        "Vyber normalizační režim:",
        ["Žádná", "Z-score", "Min-Max"]
    )

    df_plot = df.copy()

    # ---- aplikace normalizace ----
    if y_cols:
        if scale_mode == "Z-score":
            for col in y_cols:
                df_plot[col] = (df_plot[col] - df_plot[col].mean()) / df_plot[col].std()

        elif scale_mode == "Min-Max":
            for col in y_cols:
                min_val = df_plot[col].min()
                max_val = df_plot[col].max()
                df_plot[col] = (df_plot[col] - min_val) / (max_val - min_val)

        # ---- vykreslení ----
        fig = px.line(df_plot, x=time_col, y=y_cols)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Vyber alespoň jeden signál.")
else:
    st.info("Nahraj CSV soubor s časovými řadami.")

