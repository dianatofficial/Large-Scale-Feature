import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class RawDocument(BaseModel):
    document_id: str
    content: str
    source: Optional[str] = "unknown"
    category: Optional[str] = "general"
    timestamp: Optional[str] = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CleanedDocument(BaseModel):
    document_id: str
    cleaned_content: str
    content_hash: str
    char_count: int
    word_count: int
    lang: str = "en"
    source: Optional[str] = "unknown"
    category: Optional[str] = "general"
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    total_chunks: int
    chunk_content: str
    token_count: int
    start_char: int
    end_char: int
    source: Optional[str] = "unknown"
    category: Optional[str] = "general"
    metadata: Dict[str, Any] = Field(default_factory=dict)

class VectorPayload(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    chunk_content: str
    vector_fp32: Optional[List[float]] = None
    vector_int8: Optional[List[int]] = None
    scale_factor: float = 1.0
    zero_point: int = 0
    l2_norm: float = 1.0
    partition_id: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BatchStageMetrics(BaseModel):
    stage_name: str
    input_records: int
    output_records: int
    duration_seconds: float
    throughput_records_per_sec: float
    memory_used_mb: float = 0.0

class PipelineAuditReport(BaseModel):
    job_id: str
    status: str
    started_at: str
    completed_at: str
    total_duration_seconds: float
    total_input_records: int
    total_output_vectors: int
    overall_throughput_records_per_sec: float
    stage_metrics: List[BatchStageMetrics]
    quantization_mode: str
    vector_dimension: int
    num_partitions: int
    storage_sink: str
    errors: List[str] = Field(default_factory=list)

def get_spark_raw_schema():
    """Returns the PySpark StructType for raw document ingestion."""
    try:
        from pyspark.sql.types import StructType, StructField, StringType, MapType
        return StructType([
            StructField("document_id", StringType(), False),
            StructField("content", StringType(), False),
            StructField("source", StringType(), True),
            StructField("category", StringType(), True),
            StructField("timestamp", StringType(), True),
            StructField("metadata", MapType(StringType(), StringType()), True)
        ])
    except ImportError:
        return None

def get_spark_vector_schema():
    """Returns the PySpark StructType for partitioned vector feature records."""
    try:
        from pyspark.sql.types import (
            StructType, StructField, StringType, IntegerType, FloatType, 
            ArrayType, BinaryType, MapType
        )
        return StructType([
            StructField("chunk_id", StringType(), False),
            StructField("document_id", StringType(), False),
            StructField("chunk_index", IntegerType(), False),
            StructField("chunk_content", StringType(), False),
            StructField("vector_fp32", ArrayType(FloatType()), True),
            StructField("vector_int8", BinaryType(), True),
            StructField("scale_factor", FloatType(), False),
            StructField("zero_point", IntegerType(), False),
            StructField("l2_norm", FloatType(), False),
            StructField("partition_id", IntegerType(), False),
            StructField("metadata", MapType(StringType(), StringType()), True)
        ])
    except ImportError:
        return None
