import numpy as np
import streamlit as st
from app.core.model import HONU, SlidingHONU
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.utils.eta import ETAEstimator, format_eta


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
        return np.array(x_vec)

    model_type = st.selectbox("Select Model:", ["HONU", "SlidingHONU"])
    st.session_state['model_type'] = model_type

    # ----------------- STOP / RESUME -----------------
    col1, col2 = st.columns(2)
    with col1:
        stop_training = st.button("🛑 Stop training")
    with col2:
        resume_training = st.button("▶️ Resume training")

    progress = st.progress(0.0)
    status = st.empty()

    if model_type == "HONU":
        degree = st.slider("Polynomial degree (1-3):", 1, 3, 1)
        method = st.selectbox("Learning method:", ["LMS", "NGD", "RLS"])
        mu = st.slider("Learning rate μ:", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")

        n_inputs = sum(lags[col] for col in y_cols if col != target_signal)
        model = HONU(degree, n_inputs, mu, method)

    else:
        degree = st.slider("Polynomial degree (1-3):", 1, 3, 1)
        window_size = st.slider("Sliding window size M:", 10, 200, 50, 10)
        lam = st.slider("Regularization λ:", 0.0, 0.1, 0.01, 0.001, format="%.3f")

        n_inputs = sum(lags[col] for col in y_cols if col != target_signal)
        model = SlidingHONU(degree, n_inputs, window_size, lam)

    st.session_state['model'] = model


    if "train_k" not in st.session_state or not resume_training:
        st.session_state.train_k = max_lag

    start_k = st.session_state.train_k
    y_pred = np.zeros(N)
    error = np.zeros(N)
    wall = np.zeros((N, model.n_weights))
    delta_wall = np.zeros((N, model.n_weights))
    errors_list = []
    eta_est = ETAEstimator()

    LM_STEP = 10


    with st.spinner("Training model..."):
        for k in range(start_k, N):
            eta_est.tick()

            if stop_training:
                st.session_state.train_k = k
                st.session_state.model = model
                st.warning(f"Training paused at step {k}")
                break

            x = build_input_vector(k)

            if model_type == "HONU":
                err = model.update(x, df[target_signal].values[k])
            else:  # SlidingHONU
                model.add_sample(x, df[target_signal].values[k])
                if k % LM_STEP == 0:
                    err, delta_w = model.update_LM()
                else:
                    err = error[k-1] if k > 0 else 0

            y_pred[k] = model.predict(x)
            error[k] = err
            errors_list.append(err)

            wall[k] = model.w
            if model_type == "HONU":
                delta_wall[k] = model.w - wall[k-1] if k>0 else model.w
            else:
                delta_wall[k] = delta_w if k % LM_STEP == 0 else delta_wall[k-1] if k>0 else np.zeros(model.n_weights)

            if k % 10 == 0:
                eta_val = eta_est.eta(N - k)
                progress.progress(k / (N - 1))
                status.text(f"{model_type} step {k}/{N} | ETA: {format_eta(eta_val)}")

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
        title=f"{model_type} prediction ({method if model_type=='HONU' else 'Sliding LM'})"
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
