import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import networkx as nx


def page5_modes_graph():
    st.title("5) Modes and Knowledge Graph")

    if 'X_model' not in st.session_state:
        st.warning("No embedding available. Please complete Step 1 first.")
        return

    X_embed = st.session_state['X_model']


    st.subheader("Clustering of embeddings")
    n_clusters = st.slider("Number of modes/clusters:", 2, 10, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    try:
        labels = kmeans.fit_predict(X_embed)
    except Exception as e:
        st.error(f"Clustering failed: {e}")
        return

    st.write(f"Cluster labels: {np.unique(labels)}")
    if len(np.unique(labels)) < 2:
        st.warning("Not enough distinct clusters for silhouette or graph analysis.")
        return


    try:
        sil = silhouette_score(X_embed, labels)
        st.write(f"Silhouette score: {sil:.3f}")
    except Exception as e:
        st.warning(f"Cannot compute silhouette: {e}")


    st.subheader("Statistics per cluster")
    stats = []
    X_df = pd.DataFrame(X_embed)
    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        cluster_data = X_df.iloc[idx]
        stats.append({
            "Cluster": f"R{lbl}",
            "Size": len(idx),
            "Mean LE": float(np.mean(np.sum((cluster_data - kmeans.cluster_centers_[lbl]) ** 2, axis=1))),
            "Std LE": float(np.std(np.sum((cluster_data - kmeans.cluster_centers_[lbl]) ** 2, axis=1)))
        })
    st.dataframe(pd.DataFrame(stats))

    # --- Knowledge Graph ---
    st.subheader("Knowledge Graph (mode transitions)")
    # Simulace přechodů: labels → (src, dst)
    transitions = {}
    for i in range(1, len(labels)):
        src, dst = labels[i - 1], labels[i]
        transitions[(src, dst)] = transitions.get((src, dst), 0) + 1

    G = nx.DiGraph()
    for lbl in np.unique(labels):
        G.add_node(lbl)
    for (src, dst), cnt in transitions.items():
        G.add_edge(src, dst, weight=cnt)

    pos = nx.spring_layout(G, seed=42)


    edge_x, edge_y = [], []
    for src, dst in G.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='gray'),
                            hoverinfo='none', mode='lines')

    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"R{node}")
    node_trace = go.Scatter(
        x=node_x, y=node_y, text=node_text,
        mode='markers+text', textposition="bottom center",
        hoverinfo='text', marker=dict(size=20, color='lightblue')
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title="Mode Knowledge Graph", showlegend=False)
    st.plotly_chart(fig, width='stretch')


def main():
    page5_modes_graph()
    st.info("Cluster statistics and knowledge graph ready.")


if __name__ == "__main__":
    main()
