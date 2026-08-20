import streamlit as st

def render_pipeline_dag():
    """Renders high-level distributed execution DAG graph."""
    dag_code = """
    graph LR
        A[Raw Corpus Ingest<br/>JSONL / Parquet] --> B[Unicode & Lexical Clean<br/>PySpark Pandas UDF]
        B --> C[Exact Deduplication<br/>SHA-256 / Content Hash]
        C --> D[Recursive Chunking<br/>Token-Aware Explode]
        D --> E[Batch Embedding Gen<br/>Dense Vectors dim=384]
        E --> F[Vectorized L2 Normalization<br/>Arrow Vector SIMD]
        F --> G[INT8 Scalar Quantization<br/>4x Compression]
        G --> H[Hash Sharding & Staging<br/>Parquet / Qdrant Sink]

        style A fill:#1E293B,stroke:#38BDF8,stroke-width:2px,color:#F8FAFC
        style B fill:#1E293B,stroke:#38BDF8,stroke-width:2px,color:#F8FAFC
        style C fill:#1E293B,stroke:#38BDF8,stroke-width:2px,color:#F8FAFC
        style D fill:#1E293B,stroke:#38BDF8,stroke-width:2px,color:#F8FAFC
        style E fill:#1E293B,stroke:#818CF8,stroke-width:2px,color:#F8FAFC
        style F fill:#1E293B,stroke:#818CF8,stroke-width:2px,color:#F8FAFC
        style G fill:#1E293B,stroke:#34D399,stroke-width:2px,color:#F8FAFC
        style H fill:#1E293B,stroke:#34D399,stroke-width:2px,color:#F8FAFC
    """
    st.markdown(f"```mermaid\n{dag_code}\n```")
