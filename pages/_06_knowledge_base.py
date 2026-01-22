import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path


def page6_data_modes():
    st.title("6) Data Modes Processing and Knowledge Base")

    if 'model' not in st.session_state or 'X_model' not in st.session_state:
        st.warning("Please run Steps 1–4 first to have trained model and embeddings.")
        return

    model = st.session_state['model']
    X_model = st.session_state['X_model']
    df = st.session_state['raw_data']
    y_cols = st.session_state['y_cols']
    lags = st.session_state['lags']
    target_signal = st.session_state.get('target_signal', y_cols[0])


    st.subheader("Select Data Record")
    record_type = st.radio("Record type:", ["Single long record", "Multiple records"])

    if record_type == "Single long record":
        records = [("Patient1", df)]
    else:
        st.info("Upload multiple CSVs. Each will be processed separately.")
        uploaded_files = st.file_uploader("Upload CSV files:", type=["csv"], accept_multiple_files=True)
        records = []
        for f in uploaded_files:
            try:
                df_multi = pd.read_csv(f)
            except:
                df_multi = pd.read_csv(f, sep=';')
            records.append((f.name, df_multi))

    st.subheader("Process Records and Save Data")

    kb_dir = Path("knowledge_base")
    kb_dir.mkdir(exist_ok=True)

    for name, df_rec in records:
        st.write(f"Processing {name} ...")

        max_lag = max(lags.values())
        N = len(df_rec)
        le = np.zeros(N)
        y_pred = np.zeros(N)
        for k in range(max_lag, N):
            x_vec = []
            for col in y_cols:
                if col == target_signal:
                    continue
                lag = lags[col]
                x_vec.extend(df_rec[col].values[k - lag:k])
            x_vec = np.array(x_vec)
            y_pred[k] = model.predict(x_vec)
            le[k] = (df_rec[target_signal].values[k] - y_pred[k]) ** 2


        top_le_idx = np.argsort(le)[-5:]

        from pages._04_saliency_analysis import compute_saliency
        saliency = compute_saliency(model, y_cols, lags, df_rec, target_signal)
        top_saliency_idx = np.argsort(np.sum(saliency, axis=1))[-5:]

        # Save all relevant data
        record_data = {
            "config": {
                "lags": lags,
                "target_signal": target_signal,
                "model_type": st.session_state.get("model_type", "HONU"),
                "degree": st.session_state.get("degree", 1)
            },
            "inputs": df_rec.to_dict(),
            "weights": model.w.tolist(),
            "delta_weights": (model.w - model.w).tolist(),
            "LE": le.tolist(),
            "top_LE_events": top_le_idx.tolist(),
            "top_saliency_events": top_saliency_idx.tolist()
        }

        save_path = kb_dir / f"{name}_kb.json"
        with open(save_path, "w") as f:
            json.dump(record_data, f, indent=4)

        st.success(f"Record {name} saved to knowledge base with autotags.")

    st.info(
        "Knowledge base now contains configurations, inputs, weights, LE, modes, and saliency tags.\n"
        "These can be used later for fuzzy/LLM-based interpretation and analysis."
    )


def main():
    page6_data_modes()


if __name__ == "__main__":
    main()
