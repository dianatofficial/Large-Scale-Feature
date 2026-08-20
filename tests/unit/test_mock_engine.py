from src.simulation.mock_engine import VectorScaleSimulationEngine

def test_simulation_engine_end_to_end():
    engine = VectorScaleSimulationEngine(embedding_dim=384, chunk_size=64, num_partitions=4)
    raw_df = engine.generate_synthetic_corpus(num_documents=10)
    
    assert len(raw_df) == 10
    assert "content" in raw_df.columns

    final_df, audit_report, analytics = engine.run_simulation_pipeline(raw_df, quantization_mode="INT8")

    assert len(final_df) >= 10
    assert audit_report.status == "SUCCESS"
    assert audit_report.total_output_vectors == len(final_df)
    assert len(audit_report.stage_metrics) == 5
    assert "vector_matrix" in analytics
    assert analytics["vector_matrix"].shape[1] == 384
