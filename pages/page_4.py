import streamlit as st
import numpy as np
import plotly.express as px

def compute_saliency(model, y_cols, lags, df, target_signal):

    max_lag = max(lags.values())
    N = len(df)
    n_inputs = sum(lags[col] for col in y_cols if col != target_signal)
    S = np.zeros((N, n_inputs))

    def build_input_vector(k):
        x_vec = []
        for col in y_cols:
            if col == target_signal:
                continue
            lag = lags[col]
            x_vec.extend(df[col].values[k-lag:k])
        return np.array(x_vec)

    for k in range(max_lag, N):
        x = build_input_vector(k)
        eps = 1e-6
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_eps = x.copy()
            x_eps[i] += eps
            y_plus = model.predict(x_eps)
            y = model.predict(x)
            grad[i] = abs(y_plus - y) / eps
        S[k, :] = grad

    return S[max_lag:, :]

def saliency_maps_page():
    st.title("4) Saliency Maps")

    if 'model' not in st.session_state:
        st.warning("Please train a model first in Step 2.")
        return
    if 'y_cols' not in st.session_state or 'lags' not in st.session_state:
        st.warning("Signal/lags information not available. Run Step 1 first.")
        return

    model = st.session_state['model']
    df = st.session_state['raw_data']
    y_cols = st.session_state['y_cols']
    lags = st.session_state['lags']

    target_signal = st.selectbox("Select signal for saliency map:", y_cols)

    st.subheader("Compute Saliency Map")
    st.write("This may take some time depending on data size and number of features...")

    S = compute_saliency(model, y_cols, lags, df, target_signal)

    st.subheader("Saliency Heatmap")
    fig = px.imshow(
        S.T,
        labels=dict(x="Time step", y="Input features", color="Sensitivity"),
        aspect="auto",
        title=f"Saliency Map for {target_signal} (abs(∂y/∂x))"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Horizontal axis: time steps\n"
        "Vertical axis: input features (signals and lags)\n"
        "Color: absolute sensitivity of prediction w.r.t each input"
    )

def main():
    saliency_maps_page()
    st.info("Saliency maps computed from the trained model using original input vectors.")

if __name__ == "__main__":
    main()
