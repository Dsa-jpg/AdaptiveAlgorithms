import streamlit as st
import numpy as np
from app.core.model import HONU, SlidingHONU, SimpleMLP
from app.utils.model_utils import build_input_vector, initialize_storage
from app.utils.plot_utils import plot_model_training

def select_model_page():
    st.title("2) Model Selection")

    if 'y_cols' not in st.session_state or len(st.session_state['y_cols']) == 0:
        st.warning("Please select at least one signal in Step 1 first.")
        return

    df = st.session_state['raw_data']
    y_cols = st.session_state['y_cols']
    lags = st.session_state['lags']

    target_signal = st.selectbox("Which signal do you want to predict?", y_cols)
    st.session_state['target_signal'] = target_signal

    max_lag = max(lags.values())
    N = len(df)

    model_type = st.selectbox("Select Model:", ["HONU", "SlidingHONU", "MLP"])
    st.session_state['model_type'] = model_type

    n_inputs = sum(lags[col] for col in y_cols if col != target_signal)

    # ---------------- MODEL SELECTION ----------------
    if model_type == "HONU":
        degree = st.slider("Polynomial degree (1-3):", 1, 3, 1)
        method = st.selectbox("Learning method:", ["LMS", "NGD", "RLS"])
        mu = st.slider("Learning rate μ:", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")
        model = HONU(degree, n_inputs, mu, method)

    elif model_type == "SlidingHONU":
        degree = st.slider("Polynomial degree (1-3):", 1, 3, 1)
        window_size = st.slider("Sliding window size M:", 10, 200, 50, 10)
        lam = st.slider("Regularization λ:", 0.0, 0.1, 0.01, 0.001, format="%.3f")
        model = SlidingHONU(degree, n_inputs, window_size, lam)

    else:
        n_hidden = st.slider("Number of hidden neurons:", 1, 50, 10)
        lr = st.slider("Learning rate:", 0.0001, 0.1, 0.01, 0.0001, format="%.4f")
        model = SimpleMLP([n_inputs, n_hidden, 1], activations=['relu', 'linear'], lr=lr)

    st.session_state['model'] = model

    # ---------------- TRAINING ----------------
    y_pred, error, wall, delta_wall = initialize_storage(N, model_type, model)

    for k in range(max_lag, N):
        x = build_input_vector(df, y_cols, lags, target_signal, k)
        y_true = df[target_signal].values[k:k+1]

        if model_type == "HONU":
            err = model.update(x.flatten(), y_true[0])
            y_pred[k] = model.predict(x.flatten())
            delta_w = model.w - wall[k-1] if k > 0 else model.w

        elif model_type == "SlidingHONU":
            model.add_sample(x.flatten(), y_true[0])
            err = 0
            delta_w = np.zeros(wall.shape[1])
            y_pred[k] = model.predict(x.flatten())

        else:
            y_pred[k] = model.forward(x)
            err = model.backward(y_true)
            delta_w = np.concatenate([w.flatten() for w in model.weights]) - (wall[k-1] if k>0 else 0)

        error[k] = err
        wall[k] = model.w.flatten() if model_type != "MLP" else np.concatenate([w.flatten() for w in model.weights])
        delta_wall[k] = delta_w

    st.session_state['wall'] = wall
    st.session_state['delta_wall'] = delta_wall
    st.session_state['error'] = error

    fig = plot_model_training(
        np.arange(N),
        df[target_signal].values,
        y_pred,
        error,
        wall,
        delta_wall,
        title=f"{model_type} prediction"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("Model parameters and training results are stored in session_state.")


def main():
    select_model_page()


if __name__ == "__main__":
    main()
