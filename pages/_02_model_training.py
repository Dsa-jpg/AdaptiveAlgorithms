import numpy as np
import streamlit as st
from app.core.model import HONU, SlidingHONU, SimpleMLP
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def select_model():
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

    def build_input_vector(k):
        x_vec = []
        for col in y_cols:
            if col == target_signal:
                continue
            lag = lags[col]
            x_vec.extend(df[col].values[k-lag:k])
        return np.array(x_vec).reshape(1, -1)

    model_type = st.selectbox("Select Model:", ["HONU", "SlidingHONU", "MLP"])
    st.session_state['model_type'] = model_type

    n_inputs = sum(lags[col] for col in y_cols if col != target_signal)

    # ---------------- MODEL ----------------
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
    start_k = max_lag  # odstranili jsme train_k
    y_pred = np.zeros(N)
    error = np.zeros(N)

    n_weights = model.n_weights if model_type != "MLP" else sum(w.size for w in model.weights)
    wall = np.zeros((N, n_weights))
    delta_wall = np.zeros_like(wall)

    LM_STEP = 10

    for k in range(start_k, N):
        x = build_input_vector(k)
        y_true = df[target_signal].values[k:k+1]

        if model_type == "HONU":
            err = model.update(x.flatten(), y_true[0])
            y_pred[k] = model.predict(x.flatten())
            delta_w = model.w - wall[k-1] if k > 0 else model.w

        elif model_type == "SlidingHONU":
            model.add_sample(x.flatten(), y_true[0])
            if k % LM_STEP == 0:
                err, delta_w = model.update_LM()
            else:
                err = error[k-1] if k > 0 else 0
                delta_w = np.zeros(n_weights)  # <- KLÍČOVÉ

            y_pred[k] = model.predict(x.flatten())

        else:  # MLP
            y_pred[k] = model.forward(x)
            err = model.backward(y_true)
            delta_w = np.concatenate([w.flatten() for w in model.weights]) - (wall[k-1] if k>0 else 0)

        error[k] = err
        wall[k] = model.w.flatten() if model_type != "MLP" else np.concatenate([w.flatten() for w in model.weights])
        delta_wall[k] = delta_w

    st.session_state['wall'] = wall
    st.session_state['delta_wall'] = delta_wall
    st.session_state['error'] = error

    fig = plot_honu_plotly(
        np.arange(N),
        df[target_signal].values,
        y_pred,
        error,
        wall,
        delta_wall,
        title=f"{model_type} prediction"
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_honu_plotly(t, y, y_pred, error, wall, delta_wall, title="HONU"):
    n_weights = wall.shape[1]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.08,
                        subplot_titles=("Signal and Prediction",
                                        "Squared Error",
                                        "Evolution of Weights",
                                        "Evolution of Weight Change"))

    fig.add_trace(go.Scatter(x=t, y=y, mode='lines', name="y (original)", line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=y_pred, mode='lines', name="ŷ (prediction)", line=dict(color='green')), row=1, col=1)

    fig.add_trace(go.Scatter(x=t, y=error**2, mode='lines', name="Squared Error", line=dict(color='blue')), row=2, col=1)

    for i in range(n_weights):
        fig.add_trace(go.Scatter(x=t, y=wall[:, i], mode='lines', name=f"w{i+1}"), row=3, col=1)

    fig.add_trace(go.Scatter(x=t, y=np.sum(delta_wall**2, axis=1), mode='lines', name="Σ(Δw²)", line=dict(color='purple')), row=4, col=1)

    fig.update_layout(height=1000, width=1200, title_text=title, showlegend=True)
    fig.update_xaxes(title_text="Samples [k]", row=4, col=1, rangeslider_visible=True)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Error", row=2, col=1)
    fig.update_yaxes(title_text="Weight Value", row=3, col=1)
    fig.update_yaxes(title_text="Sum of Δw²", row=4, col=1)
    return fig


def main():
    select_model()
    st.info("Model parameters are stored in session_state and ready for training.")


if __name__ == "__main__":
    main()
