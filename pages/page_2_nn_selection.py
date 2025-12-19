import numpy as np
import streamlit as st
from app.core.model import HONU, SlidingHONU
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def select_model():
    st.title("2) Model Selection")

    if 'y_cols' not in st.session_state or len(st.session_state['y_cols']) == 0:
        st.warning("Please select at least one signal in Step 1 first.")
        return

    st.subheader("Select Target Signal (Output)")
    output_signal = st.selectbox("Choose signal to predict:", st.session_state['y_cols'])
    st.session_state['output_signal'] = output_signal


    input_signals = [s for s in st.session_state['y_cols'] if s != output_signal]
    st.session_state['input_signals'] = input_signals


    df = st.session_state['raw_data']
    X_data = df[input_signals].to_numpy()
    y_data = df[output_signal].to_numpy()
    t = np.arange(len(y_data))
    n_inputs = X_data.shape[1]

    model_type = st.selectbox("Select Model:", ["HONU", "SlidingHONU"])
    st.session_state['model_type'] = model_type

    if model_type == "HONU":
        degree = st.slider("Polynomial degree (1-3):", 1, 3, 1)
        method = st.selectbox("Learning method:", ["LMS", "NGD"])
        mu = st.slider("Learning rate μ:", 0.0001, 0.01, 0.01, step=0.0001, format="%.4f")

        st.session_state['degree'] = degree
        st.session_state['method'] = method
        st.session_state['mu'] = mu

        model = HONU(degree=degree, n_inputs=n_inputs, mu=mu, l_method=method)
        st.session_state['model'] = model

        y_pred = np.zeros(len(y_data))
        error = np.zeros(len(y_data))
        wall = np.zeros((len(y_data), model.n_weights))
        delta_wall = np.zeros((len(y_data), model.n_weights))

        for k in range(n_inputs, len(y_data)):
            x = X_data[k, :]
            y_pred[k] = model.predict(x)
            e = model.update(x, y_data[k])
            error[k] = e
            delta_w = model.w - wall[k - 1] if k > 0 else model.w
            wall[k] = model.w
            delta_wall[k] = delta_w

        fig = plot_honu_plotly(t, y_data, y_pred, error, wall, delta_wall, title=f"HONU ({method})")
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"HONU model initialized: degree={degree}, method={method}, μ={mu}")

    elif model_type == "SlidingHONU":
        degree = st.slider("Polynomial degree (1-3):", 1, 3, 1)
        window_size = st.slider("Sliding window size M:", 10, 200, 50, step=10)
        lam = st.slider("Regularization λ:", 0.0, 0.1, 0.01, step=0.001, format="%.3f")

        st.session_state['degree'] = degree
        st.session_state['window_size'] = window_size
        st.session_state['lam'] = lam

        model = SlidingHONU(degree=degree, n_inputs=n_inputs, window_size=window_size, lam=lam)
        st.session_state['model'] = model

        y_pred = np.zeros(len(y_data))
        error = np.zeros(len(y_data))
        wall = np.zeros((len(y_data), model.n_weights))
        delta_wall = np.zeros((len(y_data), model.n_weights))

        for k in range(n_inputs, len(y_data)):
            x = X_data[k,:].flatten()
            model.add_sample(x, y_data[k])
            y_pred[k] = model.predict(x)
            err, delta_w = model.update_LM()
            error[k] = err
            wall[k] = model.w
            delta_wall[k] = delta_w

        fig = plot_honu_plotly(t, y_data, y_pred, error, wall, delta_wall, title="Sliding-batch HONU")
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Sliding-batch HONU initialized: degree={degree}, M={window_size}, λ={lam}")


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

    fig.update_layout(
        height=1200,
        width=1400,
        title_text=title,
        showlegend=True,
        xaxis2=dict(matches='x'),
        xaxis3=dict(matches='x'),
        xaxis4=dict(matches='x')
    )

    fig.update_layout(height=900, width=1000, title_text=title, showlegend=True)
    fig.update_xaxes(title_text="Samples [k]", row=4, col=1)
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
