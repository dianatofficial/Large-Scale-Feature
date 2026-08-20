import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class SparkSettings(BaseModel):
    master: str = "local[*]"
    app_name: str = "VectorScale-Batch-Pipeline"
    driver_memory: str = "4g"
    executor_memory: str = "4g"
    executor_cores: int = 4
    default_parallelism: int = 16
    shuffle_partitions: int = 16
    arrow_max_records_per_batch: int = 10000
    kryo_buffer_max_mb: int = 512

class PipelineSettings(BaseModel):
    input_path: str = "data/raw/documents.jsonl"
    staging_path: str = "data/processed/staged_features.parquet"
    output_vector_path: str = "data/embeddings/vectors.parquet"
    batch_size: int = 2048
    max_records: Optional[int] = None

class FeatureSettings(BaseModel):
    id_column: str = "document_id"
    text_column: str = "content"
    metadata_columns: List[str] = Field(default_factory=lambda: ["source", "timestamp", "category", "author"])
    min_token_length: int = 5
    max_token_length: int = 2048
    chunk_size: int = 256
    chunk_overlap: int = 32
    deduplication_enabled: bool = True
    deduplication_method: str = "exact_hash"

class VectorSettings(BaseModel):
    model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    normalize_l2: bool = True
    quantization_mode: str = "INT8"
    clip_threshold: float = 2.5
    pca_projection_dim: Optional[int] = 3

class PartitionSettings(BaseModel):
    strategy: str = "hash"
    num_partitions: int = 16
    partition_key: str = "partition_id"

class StorageSettings(BaseModel):
    sink_type: str = "parquet"
    parquet_compression: str = "snappy"
    row_group_size_mb: int = 64
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "vectorscale_collection"
    minio_endpoint: str = "localhost:9000"
    minio_bucket: str = "vectorscale-features"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"

class LoggingSettings(BaseModel):
    level: str = "INFO"
    format: str = "json"

class AppConfig(BaseSettings):
    app_name: str = "vectorscale-engine"
    environment: str = "production"
    spark: SparkSettings = Field(default_factory=SparkSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    vector_engine: VectorSettings = Field(default_factory=VectorSettings)
    partitioning: PartitionSettings = Field(default_factory=PartitionSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    model_config = SettingsConfigDict(
        env_prefix="VECTOR_SCALE_",
        env_nested_delimiter="__",
        case_sensitive=False
    )

def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Loads configuration from a YAML file with environment variable fallback."""
    config_dict: Dict[str, Any] = {}
    if config_path and Path(config_path).is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                config_dict = loaded
    elif Path("config/base.yaml").is_file():
        with open("config/base.yaml", "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                config_dict = loaded

    return AppConfig(**config_dict)
