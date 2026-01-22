import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="AAP", layout="wide")
st.title("Adaptive Algorithms & Applications")


steps = [
    {"name": "1) Load & Embed Data", "desc": "Upload CSV, select signals, configure embeddings.", "color": "lightblue"},
    {"name": "2) Model Selection & Training", "desc": "Choose HONU/MLP, set learning parameters.", "color": "lightgreen"},
    {"name": "3) Model Monitoring", "desc": "Visualize W(k), ΔW(k), LE and projections.", "color": "lightpink"},
    {"name": "4) Saliency Maps", "desc": "Compute ∂y/∂x heatmaps for interpretability.", "color": "lightsalmon"},
    {"name": "5) Modes & Knowledge Graph", "desc": "Cluster modes and visualize knowledge graph.", "color": "lightyellow"},
    {"name": "6) Knowledge Base", "desc": "Store configurations, embeddings, weights, LE, saliency, and modes.", "color": "lightgray"}
]

fig = go.Figure()

x = list(range(len(steps)))
y = [1]*len(steps)

for i, step in enumerate(steps):
    fig.add_trace(go.Scatter(
        x=[i], y=[1],
        mode='markers+text',
        text=[step["name"]],
        textposition="bottom center",
        marker=dict(size=25, color=step["color"]),
        hovertext=step["desc"],
        hoverinfo="text"
    ))

for i in range(len(steps)-1):
    fig.add_annotation(
        x=i+1, y=1, ax=i, ay=1,
        xref='x', yref='y',
        axref='x', ayref='y',
        showarrow=True,
        arrowhead=3,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='gray'
    )

fig.update_yaxes(visible=False)
fig.update_xaxes(visible=False)
fig.update_layout(
    height=350, margin=dict(l=20, r=20, t=20, b=20),
    title_text="Project Workflow"
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("Step 1: Load & Embed Data"):
    st.markdown("""
- Upload **single/multiple CSV files** (RR, SpO₂, respiration).  
- Select signals and configure **lagged embeddings**.  
- Normalize data and optionally reduce dimensionality with **PCA**.  
""")

with st.expander("Step 2: Model Selection & Training"):
    st.markdown("""
- Choose model: **HONU**, **Sliding HONU**, or **MLP**.  
- Configure **online/sliding window parameters**.  
- Pause/resume training interactively.  
""")

with st.expander("Step 3: Model Monitoring"):
    st.markdown("""
- Visualize **weights W(k)** and **weight changes ΔW(k)**.  
- Track **Learning Entropy (LE)** over time.  
- Explore **2D projections of weights** via PCA/t-SNE.  
""")

with st.expander("Step 4: Saliency Maps"):
    st.markdown("""
- Compute **input sensitivities** (∂y/∂x).  
- Explore heatmaps to understand **LE spikes** or regime changes.  
""")

with st.expander("Step 5: Modes & Knowledge Graph"):
    st.markdown("""
- Cluster embeddings or weights to identify **system modes** (R0, R1, …).  
- Visualize **Markov transitions** and **knowledge graph** of mode dynamics.  
""")

with st.expander("Step 6: Knowledge Base"):
    st.markdown("""
- Systematically store **configurations, embeddings, weights, LE, saliency events, and modes**.  
- Autotag events and build a **growing knowledge base** for later interpretation.  
""")
