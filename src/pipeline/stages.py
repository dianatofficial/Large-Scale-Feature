import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
from pyspark.sql import DataFrame, SparkSession
from src.config.settings import AppConfig
from src.core.schemas import BatchStageMetrics
from src.spark.transformations import FeaturePipelineTransformer
from src.storage.parquet_sink import ParquetVectorSink

logger = logging.getLogger(__name__)

class PipelineStage(ABC):
    """Abstract base class representing an isolated pipeline execution stage."""

    def __init__(self, name: str, config: AppConfig):
        self.name = name
        self.config = config

    @abstractmethod
    def execute(self, spark: SparkSession, input_data: Any) -> Tuple[Any, BatchStageMetrics]:
        pass

class IngestStage(PipelineStage):
    """Reads raw document files (JSONL / Parquet) into a distributed DataFrame."""

    def __init__(self, config: AppConfig):
        super().__init__("IngestStage", config)

    def execute(self, spark: SparkSession, input_data: Any = None) -> Tuple[DataFrame, BatchStageMetrics]:
        start_time = time.time()
        input_path = self.config.pipeline.input_path
        logger.info("Reading raw input from: %s", input_path)

        if input_path.endswith(".parquet"):
            df = spark.read.parquet(input_path)
        else:
            df = spark.read.json(input_path)

        if self.config.pipeline.max_records:
            df = df.limit(self.config.pipeline.max_records)

        count = df.count()
        duration = max(time.time() - start_time, 0.001)
        throughput = count / duration

        metrics = BatchStageMetrics(
            stage_name=self.name,
            input_records=count,
            output_records=count,
            duration_seconds=round(duration, 3),
            throughput_records_per_sec=round(throughput, 2)
        )
        return df, metrics

class TransformStage(PipelineStage):
    """Executes full transformation graph (clean, dedup, chunk, embed, normalize, quantize)."""

    def __init__(self, config: AppConfig):
        super().__init__("TransformStage", config)
        self.transformer = FeaturePipelineTransformer(config)

    def execute(self, spark: SparkSession, input_data: DataFrame) -> Tuple[DataFrame, BatchStageMetrics]:
        start_time = time.time()
        input_count = input_data.count()

        transformed_df = self.transformer.transform_all(input_data)
        
        # Cache transformed DataFrame to avoid duplicate execution
        transformed_df = transformed_df.cache()
        output_count = transformed_df.count()

        duration = max(time.time() - start_time, 0.001)
        throughput = output_count / duration

        metrics = BatchStageMetrics(
            stage_name=self.name,
            input_records=input_count,
            output_records=output_count,
            duration_seconds=round(duration, 3),
            throughput_records_per_sec=round(throughput, 2)
        )
        return transformed_df, metrics

class SinkStage(PipelineStage):
    """Persists processed vectors to partitioned Parquet storage and/or Vector DB."""

    def __init__(self, config: AppConfig):
        super().__init__("SinkStage", config)
        self.parquet_sink = ParquetVectorSink(config.storage, config.partitioning)

    def execute(self, spark: SparkSession, input_data: DataFrame) -> Tuple[str, BatchStageMetrics]:
        start_time = time.time()
        record_count = input_data.count()
        output_path = self.config.pipeline.output_vector_path

        self.parquet_sink.write_dataset(input_data, output_path)

        duration = max(time.time() - start_time, 0.001)
        throughput = record_count / duration

        metrics = BatchStageMetrics(
            stage_name=self.name,
            input_records=record_count,
            output_records=record_count,
            duration_seconds=round(duration, 3),
            throughput_records_per_sec=round(throughput, 2)
        )
        return output_path, metrics
