import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.storage.vector_db_sink import FaissIndexSink

def main():
    parser = argparse.ArgumentParser(description="Construct local vector index from staged vector parquet files.")
    parser.add_argument("--parquet-dir", type=str, default="data/embeddings/vectors.parquet", help="Path to partitioned Parquet vector directory")
    parser.add_argument("--top-k", type=int, default=5, help="Number of nearest neighbors to query")
    args = parser.parse_args()

    p = Path(args.parquet_dir)
    if not p.exists():
        print(f"Error: Parquet directory not found at {args.parquet_dir}")
        sys.exit(1)

    print(f"Loading partitioned vectors from {args.parquet_dir}...")
    try:
        df = pd.read_parquet(args.parquet_dir)
    except Exception as exc:
        print(f"Error loading parquet: {exc}")
        sys.exit(1)

    print(f"Loaded {len(df)} vector chunks.")
    if "vector_fp32" not in df.columns:
        print("Column vector_fp32 not found in dataset.")
        sys.exit(1)

    vectors = np.vstack(df["vector_fp32"].values)
    ids = df["chunk_id"].tolist()
    payloads = df[["document_id", "category", "chunk_content"]].to_dict(orient="records")

    sink = FaissIndexSink(vector_dim=vectors.shape[1])
    sink.add_vectors(ids=ids, vectors=vectors, payloads=payloads)
    print(f"Index built successfully with {len(ids)} vectors.")

    # Run sample test query using first vector
    sample_query = vectors[0]
    results = sink.search(sample_query, top_k=args.top_k)

    print(f"\n--- Top {args.top_k} Nearest Neighbors for Benchmark Query ---")
    for r in results:
        print(f"ID: {r['id']} | Cosine Score: {r['score']:.4f} | Category: {r['payload'].get('category')}")

if __name__ == "__main__":
    main()
