"""
visualize.py
Helpers for embedding visualization in Streamlit
"""

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import streamlit as st

def plot_embeddings(embeddings, labels):
    """2D t-SNE plot of embeddings"""
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots()
    ax.scatter(reduced[:, 0], reduced[:, 1])
    for i, label in enumerate(labels):
        ax.annotate(label, (reduced[i, 0], reduced[i, 1]))
    st.pyplot(fig)
