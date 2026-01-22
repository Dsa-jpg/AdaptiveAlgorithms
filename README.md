# Adaptive Algorithms & Applications (AAP)


![Python](https://img.shields.io/badge/python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/streamlit-app-red)
![Project](https://img.shields.io/badge/project-university-lightblue)
![Status](https://img.shields.io/badge/status-research-orange)

Interactive Streamlit application for **adaptive prediction, regime detection, and explainability**
in time-series systems.

The project integrates **online learning**, **model monitoring**, and **interpretable AI**
with systematic accumulation of knowledge.
 

Can be run by following command '''streamlit run main.py'''
---

## Main Features

- Adaptive prediction from time-series data (RR, SpO₂, respiration, etc.)
- Online and sliding-window learning
- Models: HONU (polynomial) and MLP
- Embedding selection using Conditional Entropy (CE)
- Learning Entropy (LE) for regime change detection
- Gradient-based saliency maps for explainability
- Clustering of model weights into regimes
- Knowledge graph of regime transitions
- Persistent knowledge base with automatic tagging

---

## Workflow

1. Load data and configure embeddings (lags, PCA/SVD)
2. Select model and learning method
3. Monitor weights, LE, and projections
4. Analyze saliency and explain regime changes
5. Discover regimes and transitions
6. Store results in the knowledge base

---

## User Interface

- Built with **Streamlit**
- Multipage layout with sidebar navigation
- Interactive workflow visualization (Plotly)
- Project status summary on the main page

---