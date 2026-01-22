import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from app.utils.monitor_utils import compute_LE_window, project_weights_2D

def monitor_model_page():
    st.title("3) Model Monitoring")

    if 'model' not in st.session_state:
        st.info("Please train a model first in Step 2.")
        return
    if 'wall' not in st.session_state or 'delta_wall' not in st.session_state or 'error' not in st.session_state:
        st.info("No training data available for monitoring. Run Step 2 first.")
        return

    wall = st.session_state['wall']
    delta_wall = st.session_state['delta_wall']
    error = st.session_state['error']

    # ---- LE over window ----
    st.subheader("Learning Error (LE) over sliding window")
    window_size = st.slider("Window size for LE estimation:", min_value=10, max_value=200, value=50, step=10)
    LE_window = compute_LE_window(error, window_size)
    if LE_window.size > 0:
        st.line_chart(LE_window)
    else:
        st.info("Not enough data points to compute LE for this window size.")

    # ---- Weights W(k) ----
    st.subheader("Evolution of Weights W(k)")
    fig_w = go.Figure()
    for i in range(wall.shape[1]):
        fig_w.add_trace(go.Scatter(y=wall[:,i], mode='lines', name=f"w{i+1}"))
    fig_w.update_layout(height=400, width=900, title="Weights W(k)", xaxis_title="Samples [k]", yaxis_title="Weight Value")
    st.plotly_chart(fig_w)

    # ---- Weight changes ΔW(k) ----
    st.subheader("Evolution of Weight Changes ΔW(k)")
    fig_dw = go.Figure()
    fig_dw.add_trace(go.Scatter(y=np.sum(delta_wall**2, axis=1), mode='lines', name="ΣΔw²", line=dict(color='purple')))
    fig_dw.update_layout(height=400, width=900, title="Weight Changes ΔW(k)", xaxis_title="Samples [k]", yaxis_title="Sum of Δw²")
    st.plotly_chart(fig_dw)

    # ---- 2D projection ----
    st.subheader("2D Projection of Weights")
    method = st.selectbox("Projection method:", ["PCA", "t-SNE"])
    proj = project_weights_2D(wall, method)
    if proj.size > 0:
        fig_proj = px.scatter(x=proj[:,0], y=proj[:,1], title=f"2D Projection of Weights ({method})", labels={"x":"Dim 1", "y":"Dim 2"})
        st.plotly_chart(fig_proj)
    else:
        st.info("Not enough data points for 2D projection.")


def main():
    monitor_model_page()
    st.info("Monitoring metrics are computed from the trained model and ready for analysis.")


if __name__ == "__main__":
    main()
