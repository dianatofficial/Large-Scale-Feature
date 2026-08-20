import os
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from src.storage.parquet_sink import ParquetVectorSink
from src.config.settings import StorageSettings, PartitionSettings

def test_parquet_metadata_reader(temp_dir):
    out_dir = os.path.join(temp_dir, "test_parquet_sink")
    os.makedirs(out_dir, exist_ok=True)
    
    # Create sample pyarrow table
    df = pd.DataFrame({
        "chunk_id": [f"c_{i}" for i in range(50)],
        "vector_dim": [384] * 50,
        "partition_id": [i % 4 for i in range(50)]
    })
    table = pa.Table.from_pandas(df)
    pq.write_to_dataset(table, root_path=out_dir, partition_cols=["partition_id"])

    summary = ParquetVectorSink.read_dataset_summary(out_dir)
    assert summary["total_records"] == 50
    assert "chunk_id" in summary["schema_fields"]
