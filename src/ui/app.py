import os
import sys
from pathlib import Path
import streamlit as st

# Bootstrap root path for zero-dependency imports
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.simulation.mock_engine import VectorScaleSimulationEngine
from src.ui.tabs.live_pipeline import render_live_pipeline_tab
from src.ui.tabs.embedding_sandbox import render_embedding_sandbox_tab
from src.ui.tabs.vector_inspector import render_vector_inspector_tab
from src.ui.tabs.cluster_monitor import render_cluster_monitor_tab

st.set_page_config(
    page_title="VectorScale | Distributed Feature Engineering Engine",
    page_icon="?",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Theme CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .badge-live {
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-sim {
        background: linear-gradient(135deg, #6366F1, #4F46E5);
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("? VectorScale Engine")
st.sidebar.caption("High-Throughput Distributed Feature Pipeline")

cluster_mode = st.sidebar.selectbox(
    "Execution Engine Mode",
    ["Standalone In-Memory Sandbox (Cloud Ready)", "Apache Spark Cluster (Live Mode)"],
    index=0
)

st.sidebar.divider()
st.sidebar.markdown("### Pipeline Hyperparameters")
num_docs = st.sidebar.slider("Batch Document Volume", min_value=20, max_value=1000, value=120, step=20)
embedding_dim = st.sidebar.selectbox("Vector Dimensions", [384, 768, 1536], index=0)
chunk_size = st.sidebar.slider("Token Chunk Size", min_value=64, max_value=512, value=128, step=32)
chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=8, max_value=64, value=16, step=8)
quantization_mode = st.sidebar.selectbox("Vector Quantization", ["INT8", "FP32"], index=0)
num_partitions = st.sidebar.select_slider("Partition Shards", options=[2, 4, 8, 16, 32], value=8)

st.sidebar.divider()
st.sidebar.markdown("### Target Vector Sink")
st.sidebar.code("Format: Parquet (Snappy)\nSharding: MurmurHash3\nVector Dim: " + str(embedding_dim) + "\nStorage: Local / MinIO S3", language="yaml")

# Initialize Engine
sim_engine = VectorScaleSimulationEngine(
    embedding_dim=embedding_dim,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    num_partitions=num_partitions,
    clip_threshold=2.5
)

# Header
col_title, col_badge = st.columns([4, 1])
with col_title:
    st.markdown('<div class="main-header">? VectorScale Distributed Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Large-Scale Batch Feature Engineering, Arrow SIMD Token Chunking & Vector Database Staging</div>', unsafe_allow_html=True)
with col_badge:
    st.write("")
    if "Spark" in cluster_mode:
        st.markdown('<span class="badge-live">? LIVE SPARK CLUSTER</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-sim">?? ZERO-DEP SANDBOX</span>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "?? Batch Transformation Pipeline",
    "?? Interactive Vector & Quantization Sandbox",
    "?? Partition & Similarity Search",
    "??? Distributed Cluster Telemetry"
])

with tab1:
    render_live_pipeline_tab(sim_engine, num_docs, quantization_mode)

with tab2:
    render_embedding_sandbox_tab(sim_engine)

with tab3:
    render_vector_inspector_tab(sim_engine)

with tab4:
    render_cluster_monitor_tab()
