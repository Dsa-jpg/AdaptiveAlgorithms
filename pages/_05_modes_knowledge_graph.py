import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def page5_modes_graph():
    st.title("5) Modes and Knowledge Graph")

    if 'X_model' not in st.session_state:
        st.warning("No embedding available. Please complete Step 1 first.")
        return

    X = st.session_state['X_model']
    st.write("Embedding shape:", X.shape)


    Xs = StandardScaler().fit_transform(X)


    st.subheader("Discrete system modes")
    n_clusters = st.slider("Number of modes (clusters)", 2, 10, 3)

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    labels = kmeans.fit_predict(Xs)

    st.write("Modes:", [f"R{i}" for i in range(n_clusters)])

    if len(np.unique(labels)) < 2:
        st.warning("Not enough modes for analysis.")
        return

    sil = silhouette_score(Xs, labels)
    st.caption(f"Silhouette score (time-correlated data): {sil:.3f}")


    st.subheader("Mode statistics")

    stats = []
    for k in range(n_clusters):
        idx = np.where(labels == k)[0]
        dist = np.linalg.norm(Xs[idx] - kmeans.cluster_centers_[k], axis=1)

        stats.append({
            "Mode": f"R{k}",
            "Occupancy": len(idx),
            "Occupancy %": 100 * len(idx) / len(labels),
            "Mean distance": np.mean(dist),
            "Std distance": np.std(dist)
        })

    st.dataframe(pd.DataFrame(stats))


    st.subheader("Markov transition model")

    T = np.zeros((n_clusters, n_clusters))
    for t in range(1, len(labels)):
        T[labels[t-1], labels[t]] += 1

    T = T / (T.sum(axis=1, keepdims=True) + 1e-12)

    st.write("Transition matrix P(Rᵢ → Rⱼ)")
    st.dataframe(pd.DataFrame(
        T,
        columns=[f"R{j}" for j in range(n_clusters)],
        index=[f"R{i}" for i in range(n_clusters)]
    ))


    st.subheader("Knowledge graph of mode dynamics")

    G = nx.DiGraph()
    for i in range(n_clusters):
        G.add_node(i)

    THRESH = st.slider("Transition probability threshold", 0.01, 0.3, 0.05)

    for i in range(n_clusters):
        for j in range(n_clusters):
            if T[i, j] > THRESH:
                G.add_edge(i, j, weight=T[i, j])

    pos = nx.spring_layout(G, seed=42)


    edge_x, edge_y, edge_text = [], [], []
    for i, j, d in G.edges(data=True):
        x0, y0 = pos[i]
        x1, y1 = pos[j]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_text.append(f"P(R{i}→R{j}) = {d['weight']:.2f}")

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=2),
        hoverinfo='text',
        text=edge_text
    )

    # nodes
    occ = np.bincount(labels) / len(labels)

    node_x, node_y, node_text = [], [], []
    for i in G.nodes():
        x, y = pos[i]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"R{i}<br>P={occ[i]:.2f}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="bottom center",
        hoverinfo='text',
        marker=dict(
            size=50 * occ + 10,
            color=occ,
            colorscale='Viridis',
            showscale=True
        )
    )

    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(
        title="Symbolic knowledge graph of system dynamics",
        showlegend=False,
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)


def main():
    page5_modes_graph()
    st.info("Discrete modes, Markov model and knowledge graph successfully built.")


if __name__ == "__main__":
    main()
