import streamlit as st

def render_kpi_row(total_docs: int, total_vectors: int, throughput: float, duration: float, compression_ratio: str = "4.0x (INT8)"):
    """Renders modern executive KPI cards with gradient highlights."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(label="Raw Documents Ingested", value=f"{total_docs:,}", delta="100% Validated")
    with c2:
        st.metric(label="Vector Chunks Staged", value=f"{total_vectors:,}", delta=f"{compression_ratio}")
    with c3:
        st.metric(label="Pipeline Throughput", value=f"{throughput:,.1f} docs/sec", delta="Zero-Copy Arrow")
    with c4:
        st.metric(label="End-to-End Latency", value=f"{duration:.2f}s", delta="Parallel Execution")
