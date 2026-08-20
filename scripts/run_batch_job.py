import argparse
import json
import logging
import sys
from pathlib import Path

# Bootstrap sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.settings import load_config
from src.pipeline.runner import PipelineRunner
from config.logging_config import configure_logger

def main():
    parser = argparse.ArgumentParser(description="Execute VectorScale distributed batch feature engineering pipeline.")
    parser.add_argument("--config", type=str, default="config/local.yaml", help="Path to YAML configuration file")
    parser.add_argument("--input", type=str, default=None, help="Override input data path")
    parser.add_argument("--output", type=str, default=None, help="Override output vector parquet path")
    parser.add_argument("--quantization", type=str, default=None, help="Override quantization mode (INT8, FP32)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.input:
        cfg.pipeline.input_path = args.input
    if args.output:
        cfg.pipeline.output_vector_path = args.output
    if args.quantization:
        cfg.vector_engine.quantization_mode = args.quantization

    logger = configure_logger(name="vectorscale.cli", level=cfg.logging.level, log_format=cfg.logging.format)
    logger.info("Initializing VectorScale Batch Runner: Config=%s, Sink=%s", args.config, cfg.storage.sink_type)

    runner = PipelineRunner(cfg)
    report = runner.run()

    print("\n" + "="*80)
    print(f"VectorScale Job Summary [{report.job_id}] - Status: {report.status}")
    print(f"Total Input Records:     {report.total_input_records:,}")
    print(f"Total Vectors Staged:    {report.total_output_vectors:,}")
    print(f"Total Duration:          {report.total_duration_seconds:.2f}s")
    print(f"Overall Throughput:      {report.overall_throughput_records_per_sec:,.1f} records/sec")
    print(f"Quantization:            {report.quantization_mode} (Dim={report.vector_dimension})")
    print("="*80)

    if report.status != "SUCCESS":
        sys.exit(1)

if __name__ == "__main__":
    main()
