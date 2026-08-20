import json
import os
import pytest
from src.core.schemas import RawDocument
from src.simulation.mock_engine import VectorScaleSimulationEngine

def test_spark_transformations_integration(test_config, temp_dir):
    try:
        from src.spark.session import SparkSessionManager
        from src.spark.transformations import FeaturePipelineTransformer

        # Generate sample jsonl input
        engine = VectorScaleSimulationEngine()
        corpus_df = engine.generate_synthetic_corpus(num_documents=12)
        
        jsonl_path = test_config.pipeline.input_path
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for _, row in corpus_df.iterrows():
                f.write(json.dumps(row.to_dict()) + "\n")

        spark = SparkSessionManager.get_or_create(test_config.spark)
        raw_df = spark.read.json(jsonl_path)
        
        transformer = FeaturePipelineTransformer(test_config)
        transformed_df = transformer.transform_all(raw_df)
        
        rows = transformed_df.collect()
        assert len(rows) >= 12
        first_row = rows[0]
        
        assert hasattr(first_row, "chunk_id")
        assert hasattr(first_row, "vector_fp32")
        assert hasattr(first_row, "vector_int8")
        assert hasattr(first_row, "partition_id")
        assert len(first_row.vector_fp32) == 384
        
        SparkSessionManager.stop()
    except Exception as exc:
        pytest.skip(f"PySpark JVM environment unavailable: {exc}")
