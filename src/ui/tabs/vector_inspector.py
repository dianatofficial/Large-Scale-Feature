import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from src.core.vector_ops import l2_normalize

def render_vector_inspector_tab(sim_engine):
    st.markdown("### Vector Partitioning & Similarity Search Engine")
    st.caption("Inspect partition sharding distribution and test real-time K-Nearest Neighbor semantic vector queries.")

    data = st.session_state.get("last_run_data")
    if not data:
        st.warning("Please trigger batch execution on the Live Pipeline tab first.")
        return

    final_df = data["final_df"]
    vector_matrix = data["analytics"]["vector_matrix"]

    col_chart, col_query = st.columns([1, 1])

    with col_chart:
        st.markdown("#### Partition Distribution (Balanced Sharding)")
        part_counts = final_df["partition_id"].value_counts().reset_index()
        part_counts.columns = ["Partition ID", "Vector Count"]
        part_counts["Partition ID"] = part_counts["Partition ID"].apply(lambda p: f"Partition #{p}")

        fig = px.bar(
            part_counts,
            x="Partition ID",
            y="Vector Count",
            color="Vector Count",
            color_continuous_scale="Blues",
            template="plotly_dark",
            title="Vector Distribution Across Cluster Shards"
        )
        fig.update_layout(margin=dict(l=10, r=10, b=20, t=40))
        st.plotly_chart(fig, use_container_width=True)

    with col_query:
        st.markdown("#### Live Semantic Search Test (Cosine / Dot Product)")
        query_text = st.text_input("Enter Search Query:", "Distributed consensus and Raft protocol partition recovery")
        top_k = st.slider("Top-K Nearest Neighbors", min_value=1, max_value=10, value=3)

        if query_text:
            query_vec = sim_engine.generate_dense_embedding(query_text)
            norm_q = l2_normalize(query_vec)
            
            # Pairwise cosine score
            scores = np.dot(vector_matrix, norm_q)
            top_indices = np.argsort(scores)[::-1][:top_k]

            st.markdown(f"**Top {top_k} Vector Matches:**")
            for rank, idx in enumerate(top_indices, 1):
                row = final_df.iloc[idx]
                score = scores[idx]
                st.markdown(f"**#{rank} | Score: `{score:.4f}` | Category: `{row['category']}`**")
                st.caption(f"Chunk ID: `{row['chunk_id']}` | Partition: `#{row['partition_id']}`")
                st.markdown(f"> *{row['chunk_content'][:200]}...*")
