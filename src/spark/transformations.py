import logging
from typing import Optional
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from src.config.settings import AppConfig
from src.spark.udfs import (
    clean_text_udf,
    content_hash_udf,
    chunk_text_udf,
    generate_deterministic_embedding_udf,
    vector_l2_normalize_udf,
    vector_quantize_int8_udf,
    compute_partition_id_udf,
)

logger = logging.getLogger(__name__)

class FeaturePipelineTransformer:
    """Declarative transformation pipeline executing distributed batch operations on PySpark DataFrames."""

    def __init__(self, config: AppConfig):
        self.config = config

    def stage_clean_and_hash(self, df: DataFrame) -> DataFrame:
        """Cleans raw text column and adds deterministic content hash."""
        text_col = self.config.features.text_column
        id_col = self.config.features.id_column
        logger.info("Executing clean & hash stage on column: %s", text_col)

        return (
            df.filter(F.col(text_col).isNotNull() & (F.length(F.trim(F.col(text_col))) > 0))
            .withColumn("cleaned_content", clean_text_udf(F.col(text_col)))
            .withColumn("content_hash", content_hash_udf(F.col("cleaned_content")))
            .withColumn("char_count", F.length(F.col("cleaned_content")))
            .withColumn("word_count", F.size(F.split(F.col("cleaned_content"), r"\s+")))
            .filter(F.col("word_count") >= self.config.features.min_token_length)
        )

    def stage_deduplicate(self, df: DataFrame) -> DataFrame:
        """Performs distributed exact deduplication across Spark partitions using content hash."""
        if not self.config.features.deduplication_enabled:
            return df
        logger.info("Executing exact deduplication on content_hash")
        return df.dropDuplicates(["content_hash"])

    def stage_chunk_and_explode(self, df: DataFrame) -> DataFrame:
        """Splits normalized document content into chunk arrays and explodes them into individual rows."""
        id_col = self.config.features.id_column
        logger.info("Executing chunk & explode stage (size=%d, overlap=%d)", 
                    self.config.features.chunk_size, self.config.features.chunk_overlap)

        chunked_df = df.withColumn("chunk_array", chunk_text_udf(F.col("cleaned_content")))
        
        # Explode with positional index
        exploded_df = (
            chunked_df.select(
                F.col(id_col),
                F.col("source"),
                F.col("category"),
                F.posexplode(F.col("chunk_array")).alias("chunk_index", "chunk_content")
            )
            .withColumn("chunk_id", F.concat(F.col(id_col), F.lit("_chk_"), F.col("chunk_index")))
            .withColumn("token_count", F.size(F.split(F.col("chunk_content"), r"\s+")))
            .filter(F.col("token_count") > 0)
        )
        return exploded_df

    def stage_embed_and_normalize(self, df: DataFrame) -> DataFrame:
        """Generates high-dimensional vector embeddings and applies L2 normalization."""
        logger.info("Executing embedding & L2 normalization stage (dim=%d)", 
                    self.config.vector_engine.embedding_dim)

        embedded_df = df.withColumn(
            "vector_fp32", 
            generate_deterministic_embedding_udf(F.col("chunk_content"))
        )

        if self.config.vector_engine.normalize_l2:
            embedded_df = embedded_df.withColumn(
                "vector_fp32", 
                vector_l2_normalize_udf(F.col("vector_fp32"))
            )

        return embedded_df

    def stage_quantize_and_shard(self, df: DataFrame) -> DataFrame:
        """Applies INT8 scalar quantization, assigns partition keys, and structures schema for vector sink."""
        logger.info("Executing quantization (mode=%s) & sharding stage (num_partitions=%d)",
                    self.config.vector_engine.quantization_mode, self.config.partitioning.num_partitions)

        quantized_df = (
            df.withColumn("vector_int8", vector_quantize_int8_udf(F.col("vector_fp32")))
            .withColumn("scale_factor", F.lit(float(self.config.vector_engine.clip_threshold / 127.0)))
            .withColumn("zero_point", F.lit(0))
            .withColumn("l2_norm", F.lit(1.0))
            .withColumn("partition_id", compute_partition_id_udf(F.col("chunk_id")))
        )
        return quantized_df

    def transform_all(self, raw_df: DataFrame) -> DataFrame:
        """Executes end-to-end transformation chain from raw text to partition-ready vector records."""
        cleaned = self.stage_clean_and_hash(raw_df)
        deduped = self.stage_deduplicate(cleaned)
        chunked = self.stage_chunk_and_explode(deduped)
        embedded = self.stage_embed_and_normalize(chunked)
        sharded = self.stage_quantize_and_shard(embedded)
        return sharded
