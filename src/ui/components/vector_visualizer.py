import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.core.vector_ops import project_pca

def render_vector_scatter_3d(vectors: np.ndarray, categories: list, title: str = "3D PCA Vector Space Projection"):
    """Renders interactive 3D PCA vector space scatter plot with category color-coding."""
    if vectors.shape[0] < 3:
        st.info("Insufficient vector samples for 3D PCA projection.")
        return

    proj, explained = project_pca(vectors, n_components=3)
    
    fig = px.scatter_3d(
        x=proj[:, 0],
        y=proj[:, 1],
        z=proj[:, 2],
        color=categories,
        labels={
            "x": f"PC1 ({explained[0]*100:.1f}%)",
            "y": f"PC2 ({explained[1]*100:.1f}%)",
            "z": f"PC3 ({explained[2]*100:.1f}%)",
            "color": "Domain / Category"
        },
        title=title,
        opacity=0.85,
        template="plotly_dark"
    )
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)")
        )
    )
    st.plotly_chart(fig, use_container_width=True)

def render_cosine_heatmap(cosine_matrix: np.ndarray, labels: list, title: str = "Pairwise Cosine Similarity Heatmap"):
    """Renders pairwise cosine similarity matrix between document chunks."""
    fig = px.imshow(
        cosine_matrix,
        x=labels,
        y=labels,
        color_continuous_scale="Viridis",
        title=title,
        template="plotly_dark",
        zmin=0.0,
        zmax=1.0
    )
    fig.update_layout(margin=dict(l=20, r=20, b=20, t=40))
    st.plotly_chart(fig, use_container_width=True)

def render_quantization_comparison(fp32_vec: np.ndarray, int8_vec: np.ndarray, scale: float):
    """Visualizes bit-level differences between original FP32 and reconstructed INT8 vector dimensions."""
    dims = list(range(min(48, len(fp32_vec))))
    reconstructed = int8_vec[:len(dims)].astype(np.float32) * scale

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dims, y=fp32_vec[:len(dims)], mode="lines+markers", name="Original FP32", line=dict(color="#00D2FF", width=2)))
    fig.add_trace(go.Scatter(x=dims, y=reconstructed, mode="lines+markers", name="Reconstructed INT8 (4x Compressed)", line=dict(color="#FF5E62", width=2, dash="dot")))
    
    fig.update_layout(
        title="Vector Dimension Precision: Original FP32 vs. INT8 Reconstructed",
        xaxis_title="Vector Dimension Index",
        yaxis_title="Normalized Value",
        template="plotly_dark",
        margin=dict(l=20, r=20, b=20, t=40),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    st.plotly_chart(fig, use_container_width=True)
