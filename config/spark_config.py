from typing import Dict

def get_optimized_spark_properties(
    app_name: str = "VectorScale-Batch-Pipeline",
    driver_memory: str = "4g",
    executor_memory: str = "4g",
    executor_cores: int = 4,
    shuffle_partitions: int = 16,
    arrow_batch_size: int = 10000,
    kryo_buffer_max_mb: int = 512,
) -> Dict[str, str]:
    """Generates high-performance Spark execution configuration tuned for Arrow vectorized UDFs."""
    return {
        "spark.app.name": app_name,
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.sql.execution.arrow.pyspark.fallback.enabled": "true",
        "spark.sql.execution.arrow.maxRecordsPerBatch": str(arrow_batch_size),
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.kryoserializer.buffer.max": f"{kryo_buffer_max_mb}m",
        "spark.sql.shuffle.partitions": str(shuffle_partitions),
        "spark.default.parallelism": str(executor_cores * 4),
        "spark.driver.memory": driver_memory,
        "spark.executor.memory": executor_memory,
        "spark.sql.parquet.compression.codec": "snappy",
        "spark.sql.parquet.filterPushdown": "true",
        "spark.sql.inMemoryColumnarStorage.compressed": "true",
        "spark.rdd.compress": "true",
    }
