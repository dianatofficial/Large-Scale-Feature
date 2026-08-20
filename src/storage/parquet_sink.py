import logging
from pathlib import Path
from typing import Optional
from pyspark.sql import DataFrame
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pandas as pd
from src.config.settings import StorageSettings, PartitionSettings

logger = logging.getLogger(__name__)

class ParquetVectorSink:
    """High-performance partitioned Parquet writer optimized for vector index bulk loading."""

    def __init__(self, storage_settings: StorageSettings, partition_settings: PartitionSettings):
        self.storage_settings = storage_settings
        self.partition_settings = partition_settings

    def write_dataset(self, df: DataFrame, output_path: str, mode: str = "overwrite") -> int:
        """Writes Spark DataFrame to partitioned Parquet files on disk/object storage."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("Writing vector dataset to Parquet: Path=%s, Compression=%s, PartitionBy=%s",
                    output_path, self.storage_settings.parquet_compression, self.partition_settings.partition_key)

        (
            df.write.mode(mode)
            .partitionBy(self.partition_settings.partition_key)
            .option("compression", self.storage_settings.parquet_compression)
            .parquet(output_path)
        )
        logger.info("Successfully wrote partitioned Parquet dataset to %s", output_path)
        return 1

    @staticmethod
    def read_dataset_summary(parquet_path: str) -> dict:
        """Reads partitioned Parquet metadata and computes column statistics."""
        p = Path(parquet_path)
        if not p.exists():
            return {"error": "Path does not exist", "total_records": 0}

        try:
            dataset = ds.dataset(parquet_path, format="parquet", partitioning="hive")
            total_rows = dataset.count_rows()
            schema_names = dataset.schema.names
            files = dataset.files

            return {
                "total_records": total_rows,
                "file_count": len(files),
                "schema_fields": schema_names,
                "parquet_files": files[:10]
            }
        except Exception as exc:
            logger.warning("Error reading parquet metadata: %s", exc)
            return {"error": str(exc), "total_records": 0}
