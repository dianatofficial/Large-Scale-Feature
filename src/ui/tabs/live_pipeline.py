import streamlit as st
import pandas as pd
import plotly.express as px
from src.ui.components.metrics_card import render_kpi_row
from src.ui.components.pipeline_graph import render_pipeline_dag

def render_live_pipeline_tab(sim_engine, num_docs: int, quant_mode: str):
    st.markdown("### Distributed Batch Transformation Pipeline")
    st.caption("Executes parallel feature extraction, token chunking, and INT8 vector quantization across partitions.")

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button("Trigger Batch Transformation", type="primary", use_container_width=True)
    with col_info:
        st.info(f"Target Configuration: **{num_docs} Documents** | Model: **384-dim Dense** | Quantization: **{quant_mode}** | Partitions: **{sim_engine.num_partitions}**")

    if run_clicked or "last_run_data" not in st.session_state:
        with st.spinner("Processing distributed transformation stages..."):
            raw_corpus = sim_engine.generate_synthetic_corpus(num_documents=num_docs)
            final_df, audit_report, analytics = sim_engine.run_simulation_pipeline(raw_corpus, quantization_mode=quant_mode)
            st.session_state["last_run_data"] = {
                "final_df": final_df,
                "audit_report": audit_report,
                "analytics": analytics,
                "raw_corpus": raw_corpus
            }

    data = st.session_state.get("last_run_data")
    if not data:
        return

    report = data["audit_report"]
    final_df = data["final_df"]

    # KPIs
    render_kpi_row(
        total_docs=report.total_input_records,
        total_vectors=report.total_output_vectors,
        throughput=report.overall_throughput_records_per_sec,
        duration=report.total_duration_seconds,
        compression_ratio="4.0x (INT8)" if quant_mode == "INT8" else "1.0x (FP32)"
    )

    st.divider()

    # Stage Timings Bar Chart
    c_dag, c_chart = st.columns([1, 1])
    with c_dag:
        st.markdown("#### Execution Lineage & DAG")
        render_pipeline_dag()
    with c_chart:
        st.markdown("#### Stage Latency & Throughput Breakdown")
        stages_df = pd.DataFrame([
            {
                "Stage": m.stage_name,
                "Duration (s)": m.duration_seconds,
                "Throughput (records/s)": m.throughput_records_per_sec,
                "Output Count": m.output_records
            }
            for m in report.stage_metrics
        ])
        fig = px.bar(
            stages_df,
            x="Duration (s)",
            y="Stage",
            orientation="h",
            color="Throughput (records/s)",
            color_continuous_scale="Tealgrn",
            template="plotly_dark",
            title="Stage Duration & Processing Velocity"
        )
        fig.update_layout(margin=dict(l=10, r=10, b=20, t=40))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Staged Feature Records (Sample Output)")
    display_df = final_df[["chunk_id", "document_id", "category", "chunk_content", "scale_factor", "partition_id"]].head(10)
    st.dataframe(display_df, use_container_width=True)
