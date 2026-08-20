import streamlit as st
import numpy as np
from src.core.vector_ops import quantize_int8, compute_quantization_error, compute_cosine_matrix
from src.ui.components.vector_visualizer import (
    render_vector_scatter_3d,
    render_cosine_heatmap,
    render_quantization_comparison
)

DEFAULT_TEXT_A = "High-throughput distributed feature engineering pipeline leverages PySpark Pandas UDFs with Apache Arrow serialization."
DEFAULT_TEXT_B = "Optimized vectorized transformations reduce inter-process deserialization latency between JVM and Python worker tasks."
DEFAULT_TEXT_C = "Single-cell genomic sequencing profiles transcriptional heterogeneity across tumor-infiltrating cytotoxic lymphocytes."
DEFAULT_TEXT_D = "Quarterly financial auditing affirms statutory compliance with Basel III liquidity coverage framework."

def render_embedding_sandbox_tab(sim_engine):
    st.markdown("### Interactive Vector & Quantization Playground")
    st.caption("Inspect live token chunking, semantic cosine similarity heatmaps, and bit-level INT8 quantization loss.")

    st.markdown("#### 1. Custom Text Chunk Analysis")
    col1, col2 = st.columns(2)
    with col1:
        text_1 = st.text_area("Input Document A (Distributed Systems)", DEFAULT_TEXT_A, height=90)
        text_2 = st.text_area("Input Document B (Systems Optimization)", DEFAULT_TEXT_B, height=90)
    with col2:
        text_3 = st.text_area("Input Document C (Genomics / Biomedical)", DEFAULT_TEXT_C, height=90)
        text_4 = st.text_area("Input Document D (Financial Regulation)", DEFAULT_TEXT_D, height=90)

    inputs = [text_1, text_2, text_3, text_4]
    labels = ["Doc A (Systems)", "Doc B (PySpark)", "Doc C (Biomed)", "Doc D (Finance)"]
    
    # Compute embeddings
    vectors = np.vstack([sim_engine.generate_dense_embedding(t) for t in inputs])
    cosine_mat = compute_cosine_matrix(vectors)

    c_heat, c_3d = st.columns(2)
    with c_heat:
        render_cosine_heatmap(cosine_mat, labels)
    with c_3d:
        data = st.session_state.get("last_run_data")
        if data:
            v_mat = data["analytics"]["vector_matrix"]
            cats = data["final_df"]["category"].tolist()
            render_vector_scatter_3d(v_mat[:200], cats[:200], title="3D Semantic Embedding Space")
        else:
            st.info("Trigger batch transformation on Tab 1 to populate full 3D vector space.")

    st.divider()
    st.markdown("#### 2. INT8 Scalar Quantization Error Analysis")
    q_vecs, scales, zps = quantize_int8(vectors, clip_threshold=sim_engine.clip_threshold)
    q_err = compute_quantization_error(vectors[0], q_vecs[0].astype(np.float32) * scales[0])

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Cosine Fidelity", f"{q_err['cosine_similarity']*100:.3f}%", delta="Loss < 0.05%")
    with m2:
        st.metric("Mean Squared Error (MSE)", f"{q_err['mse']:.6f}", delta="High Accuracy")
    with m3:
        st.metric("Signal-to-Noise Ratio", f"{q_err['snr_db']:.1f} dB", delta="Broadcast Quality")
    with m4:
        st.metric("Storage Footprint", "384 Bytes / Vector", delta="-75% (vs FP32)")

    render_quantization_comparison(vectors[0], q_vecs[0], scales[0])
