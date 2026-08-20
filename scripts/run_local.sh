#!/usr/bin/env bash
set -euo pipefail

echo "=== [VectorScale] Generating Sample Dataset ==="
python scripts/generate_sample_data.py --output data/sample/documents.jsonl --num-records 500

echo "=== [VectorScale] Executing Distributed Feature Pipeline ==="
python scripts/run_batch_job.py --config config/local.yaml

echo "=== [VectorScale] Launching Streamlit Interactive Dashboard ==="
streamlit run src/ui/app.py --server.port=8501
