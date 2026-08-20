import logging
from typing import Optional
from pyspark.sql import SparkSession
from src.config.settings import SparkSettings
from config.spark_config import get_optimized_spark_properties

logger = logging.getLogger(__name__)

class SparkSessionManager:
    """Lifecycle manager for Apache Spark session with optimized Arrow configuration."""

    _session: Optional[SparkSession] = None

    @classmethod
    def get_or_create(cls, config: Optional[SparkSettings] = None) -> SparkSession:
        """Initializes or retrieves an active SparkSession configured for high-throughput vectorized operations."""
        if cls._session is not None and not cls._session.sparkContext._jsc.sc().isStopped():
            return cls._session

        settings = config or SparkSettings()
        logger.info("Initializing SparkSession: Master=%s, AppName=%s", settings.master, settings.app_name)

        builder = SparkSession.builder.master(settings.master).appName(settings.app_name)

        properties = get_optimized_spark_properties(
            app_name=settings.app_name,
            driver_memory=settings.driver_memory,
            executor_memory=settings.executor_memory,
            executor_cores=settings.executor_cores,
            shuffle_partitions=settings.shuffle_partitions,
            arrow_batch_size=settings.arrow_max_records_per_batch,
            kryo_buffer_max_mb=settings.kryo_buffer_max_mb,
        )

        for key, value in properties.items():
            builder = builder.config(key, value)

        cls._session = builder.getOrCreate()
        cls._session.sparkContext.setLogLevel("WARN")
        return cls._session

    @classmethod
    def stop(cls) -> None:
        """Gracefully stops the active SparkSession and releases executor resources."""
        if cls._session is not None:
            logger.info("Terminating SparkSession and releasing cluster executors.")
            try:
                cls._session.stop()
            except Exception as exc:
                logger.warning("Error during SparkSession stop: %s", exc)
            finally:
                cls._session = None
