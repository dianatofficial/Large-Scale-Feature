import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional
from src.config.settings import AppConfig
from src.core.schemas import BatchStageMetrics, PipelineAuditReport
from src.pipeline.stages import IngestStage, TransformStage, SinkStage
from src.spark.session import SparkSessionManager

logger = logging.getLogger(__name__)

class PipelineRunner:
    """Master orchestrator for end-to-end distributed batch feature & vector processing."""

    def __init__(self, config: AppConfig):
        self.config = config

    def run(self) -> PipelineAuditReport:
        """Executes full batch lifecycle and produces structured audit telemetry."""
        job_id = f"job_{uuid.uuid4().hex[:10]}"
        started_at = datetime.now(timezone.utc).isoformat()
        start_wall_clock = time.time()
        stage_metrics: List[BatchStageMetrics] = []
        errors: List[str] = []
        spark = None

        logger.info("Starting VectorScale Pipeline Job: %s", job_id)

        try:
            spark = SparkSessionManager.get_or_create(self.config.spark)

            # Stage 1: Ingest
            ingest_stage = IngestStage(self.config)
            raw_df, m1 = ingest_stage.execute(spark)
            stage_metrics.append(m1)

            # Stage 2: Transform
            transform_stage = TransformStage(self.config)
            vector_df, m2 = transform_stage.execute(spark, raw_df)
            stage_metrics.append(m2)

            # Stage 3: Sink
            sink_stage = SinkStage(self.config)
            output_path, m3 = sink_stage.execute(spark, vector_df)
            stage_metrics.append(m3)

            total_input = m1.input_records
            total_output = m3.output_records
            status = "SUCCESS"

        except Exception as exc:
            logger.exception("Pipeline execution failed: %s", exc)
            errors.append(str(exc))
            status = "FAILED"
            total_input = 0
            total_output = 0

        finally:
            if spark is not None:
                SparkSessionManager.stop()

        total_duration = max(time.time() - start_wall_clock, 0.001)
        overall_throughput = total_output / total_duration if total_output > 0 else 0.0
        completed_at = datetime.now(timezone.utc).isoformat()

        report = PipelineAuditReport(
            job_id=job_id,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=round(total_duration, 3),
            total_input_records=total_input,
            total_output_vectors=total_output,
            overall_throughput_records_per_sec=round(overall_throughput, 2),
            stage_metrics=stage_metrics,
            quantization_mode=self.config.vector_engine.quantization_mode,
            vector_dimension=self.config.vector_engine.embedding_dim,
            num_partitions=self.config.partitioning.num_partitions,
            storage_sink=self.config.storage.sink_type,
            errors=errors
        )

        logger.info("Pipeline Job Finished: Status=%s, TotalVectors=%d, Duration=%.2fs",
                    status, total_output, total_duration)
        return report
