import os
import shutil
import tempfile
import pytest
import numpy as np
from src.config.settings import AppConfig, SparkSettings, PipelineSettings, FeatureSettings, VectorSettings, PartitionSettings, StorageSettings

@pytest.fixture(scope="session")
def sample_raw_text():
    return (
        "<html><body><h1>Quarterly Technical Audit</h1>"
        "<p>Distributed system throughput increased by <b>45%</b> after deploying PySpark Pandas UDFs with Apache Arrow.</p>"
        "<a href='https://example.com/docs'>Read RFC documentation here</a>. "
        "Contact: engineering@vectorscale.internal. Zero-copy IPC minimizes memory serialization latency. "
        "Further metrics confirm 99.99% availability during partition failovers.</body></html>"
    )

@pytest.fixture(scope="session")
def sample_vector_matrix():
    rng = np.random.RandomState(42)
    return rng.normal(0, 1, (20, 384)).astype(np.float32)

@pytest.fixture
def temp_dir():
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture
def test_config(temp_dir):
    return AppConfig(
        app_name="vectorscale-test",
        environment="test",
        spark=SparkSettings(
            master="local[2]",
            app_name="VectorScale-Test",
            driver_memory="1g",
            executor_memory="1g",
            executor_cores=2,
            shuffle_partitions=2,
            arrow_max_records_per_batch=1000
        ),
        pipeline=PipelineSettings(
            input_path=os.path.join(temp_dir, "input.jsonl"),
            staging_path=os.path.join(temp_dir, "staging.parquet"),
            output_vector_path=os.path.join(temp_dir, "vectors.parquet"),
            batch_size=100,
            max_records=50
        ),
        features=FeatureSettings(
            chunk_size=64,
            chunk_overlap=8,
            min_token_length=2,
            deduplication_enabled=True
        ),
        vector_engine=VectorSettings(
            embedding_dim=384,
            normalize_l2=True,
            quantization_mode="INT8",
            clip_threshold=2.5
        ),
        partitioning=PartitionSettings(
            strategy="hash",
            num_partitions=2,
            partition_key="partition_id"
        ),
        storage=StorageSettings(
            sink_type="parquet",
            parquet_compression="snappy"
        )
    )
