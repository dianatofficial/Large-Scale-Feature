# VectorScale Local Execution Script for Windows PowerShell
$ErrorActionPreference = "Stop"

Write-Host "=== [VectorScale] Generating Sample Dataset ===" -ForegroundColor Cyan
python scripts/generate_sample_data.py --output data/sample/documents.jsonl --num-records 500

Write-Host "=== [VectorScale] Executing Distributed Feature Pipeline ===" -ForegroundColor Cyan
python scripts/run_batch_job.py --config config/local.yaml

Write-Host "=== [VectorScale] Launching Streamlit Interactive Dashboard ===" -ForegroundColor Green
streamlit run src/ui/app.py --server.port=8501
