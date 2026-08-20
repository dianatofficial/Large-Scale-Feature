import streamlit as st
import pandas as pd
import plotly.express as px

def render_cluster_monitor_tab():
    st.markdown("### Distributed Spark Cluster & Worker Telemetry")
    st.caption("Live monitoring of executor memory utilization, Arrow serialization throughput, and shuffle metrics.")

    # Simulated worker executor stats
    executors_data = [
        {"Executor ID": "driver", "Host": "10.0.1.10", "Cores": 4, "Memory (GB)": "3.8 / 4.0", "GC Time (ms)": 142, "Shuffle Read (MB)": 0.0, "Status": "Active"},
        {"Executor ID": "worker-1", "Host": "10.0.1.11", "Cores": 4, "Memory (GB)": "3.4 / 4.0", "GC Time (ms)": 88, "Shuffle Read (MB)": 124.5, "Status": "Active"},
        {"Executor ID": "worker-2", "Host": "10.0.1.12", "Cores": 4, "Memory (GB)": "3.6 / 4.0", "GC Time (ms)": 94, "Shuffle Read (MB)": 138.2, "Status": "Active"},
        {"Executor ID": "worker-3", "Host": "10.0.1.13", "Cores": 4, "Memory (GB)": "3.2 / 4.0", "GC Time (ms)": 76, "Shuffle Read (MB)": 115.8, "Status": "Active"},
    ]
    st.dataframe(pd.DataFrame(executors_data), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        mem_df = pd.DataFrame({
            "Executor": ["driver", "worker-1", "worker-2", "worker-3"],
            "Heap Used (MB)": [3800, 3400, 3600, 3200],
            "Off-Heap Arrow (MB)": [512, 1024, 1024, 1024]
        })
        fig = px.bar(
            mem_df,
            x="Executor",
            y=["Heap Used (MB)", "Off-Heap Arrow (MB)"],
            barmode="stack",
            title="Executor Memory Allocation (JVM vs Off-Heap Arrow)",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        task_latencies = [12, 14, 15, 13, 16, 18, 14, 15, 19, 13, 14, 16, 15, 14, 17, 13]
        fig2 = px.histogram(
            x=task_latencies,
            nbins=8,
            title="Task Duration Distribution across Partitions (ms)",
            labels={"x": "Task Duration (ms)"},
            template="plotly_dark"
        )
        st.plotly_chart(fig2, use_container_width=True)
