import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.simulation.mock_engine import VectorScaleSimulationEngine

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic multi-domain document corpus for batch feature pipeline.")
    parser.add_argument("--output", type=str, default="data/sample/documents.jsonl", help="Output JSONL file path")
    parser.add_argument("--num-records", type=int, default=1000, help="Number of records to generate")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_records} synthetic documents to {out_path}...")
    engine = VectorScaleSimulationEngine()
    df = engine.generate_synthetic_corpus(num_documents=args.num_records)

    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")

    print(f"Successfully wrote {len(df)} records to {out_path}")

if __name__ == "__main__":
    main()
