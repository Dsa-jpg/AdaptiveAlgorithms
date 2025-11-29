import streamlit as st
import pandas as pd

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
    st.write("Rychlý náhled signálů:")
    st.line_chart(df.fillna(method='ffill'))
else:
    st.info("Nahraj CSV soubor s časovými řadami. Každý sloupec = jeden signál.")
