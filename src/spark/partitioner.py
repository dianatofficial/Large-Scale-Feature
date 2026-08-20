import logging
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from src.config.settings import PartitionSettings

logger = logging.getLogger(__name__)

class VectorPartitioner:
    """Partitioning strategies for staging large vector datasets for distributed vector database loading."""

    @staticmethod
    def partition_by_hash(df: DataFrame, settings: PartitionSettings) -> DataFrame:
        """Re-partitions DataFrame evenly based on partition_id column."""
        num_partitions = settings.num_partitions
        partition_key = settings.partition_key
        logger.info("Applying Hash Partitioning: num_partitions=%d, key=%s", num_partitions, partition_key)
        return df.repartition(num_partitions, F.col(partition_key))

    @staticmethod
    def partition_by_range(df: DataFrame, settings: PartitionSettings) -> DataFrame:
        """Applies range-based partitioning for ordered vector chunk retrieval."""
        num_partitions = settings.num_partitions
        partition_key = settings.partition_key
        logger.info("Applying Range Partitioning: num_partitions=%d, key=%s", num_partitions, partition_key)
        return df.repartitionByRange(num_partitions, F.col(partition_key))
