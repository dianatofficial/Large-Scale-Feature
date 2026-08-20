import hashlib
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.core.chunker import TextChunker
from src.core.deduplicator import ExactHasher, SimHasher
from src.core.normalizer import TextNormalizer
from src.core.schemas import BatchStageMetrics, PipelineAuditReport, VectorPayload
from src.core.vector_ops import (
    compute_cosine_matrix,
    compute_quantization_error,
    dequantize_int8,
    l2_normalize,
    project_pca,
    quantize_int8,
)

DOMAINS_CORPORA = {
    "FinTech": [
        "Quarterly fiscal report indicates an aggregate recurring revenue growth of 24.8% YoY driven by enterprise subscription renewals.",
        "Quantitative risk assessment models flagged high volatility in leveraged equity swaps during Asian trading sessions.",
        "Regulatory compliance audit confirmed full adherence to Basel III capital adequacy ratios and liquidity coverage guidelines.",
        "Automated clearing house (ACH) batch latency decreased by 42ms after migrating ledger reconciliation to event-driven architectures."
    ],
    "Distributed Systems": [
        "Raft consensus protocol leader election cycle timed out due to transient asymmetric network partition across availability zones.",
        "Vectorized columnar execution using Apache Arrow reduced PySpark deserialization overhead by 78% on dense tensor matrices.",
        "Consistent hashing ring rebalanced 4096 virtual tokens upon dynamic horizontal auto-scaling of worker nodes.",
        "LSM-tree compaction policy tuned to leveled compaction strategy to minimize write amplification during burst ingestion."
    ],
    "Biomedical": [
        "Single-cell RNA sequencing revealed distinct transcriptional heterogeneity within tumor-infiltrating cytotoxic T lymphocytes.",
        "CRISPR-Cas9 guide RNA off-target cleavage was mitigated by engineered high-fidelity Cas9 variants with enhanced PAM specificity.",
        "Pharmacokinetic profiling demonstrated 84% bioavailability and an elimination half-life of 6.2 hours in mammalian models.",
        "Cryo-EM structural determination at 2.4 Angstrom resolution elucidated the allosteric inhibition mechanism of the kinase domain."
    ],
    "Legal Tech": [
        "Indemnification clause subsection 4.2 stipulates bilateral limitation of liability except in cases of gross negligence or willful misconduct.",
        "Cross-border intellectual property licensing agreement governs non-exclusive distribution across sovereign jurisdictions.",
        "Force majeure provisions were triggered following statutory supply chain embargoes under international trade sanctions.",
        "Confidentiality non-disclosure covenants shall survive the termination of this master services agreement for a term of five years."
    ]
}

class VectorScaleSimulationEngine:
    """High-fidelity standalone engine executing vectorized pipelines without external cluster dependencies."""

    def __init__(
        self,
        embedding_dim: int = 384,
        chunk_size: int = 128,
        chunk_overlap: int = 16,
        num_partitions: int = 8,
        clip_threshold: float = 2.5
    ):
        self.embedding_dim = embedding_dim
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.normalizer = TextNormalizer()
        self.simhasher = SimHasher()
        self.num_partitions = num_partitions
        self.clip_threshold = clip_threshold

    def generate_synthetic_corpus(self, num_documents: int = 100) -> pd.DataFrame:
        """Generates high-entropy realistic multi-domain text records for benchmarking."""
        records: List[Dict[str, Any]] = []
        categories = list(DOMAINS_CORPORA.keys())

        for i in range(num_documents):
            cat = categories[i % len(categories)]
            template_pool = DOMAINS_CORPORA[cat]
            base_text = template_pool[i % len(template_pool)]
            
            # Combine sentences to produce multi-paragraph document
            full_content = (
                f"### Document Record {i+1} [{cat}]\n\n"
                f"{base_text} Furthermore, additional telemetry indicates continuous monitoring. "
                f"{template_pool[(i+1) % len(template_pool)]}\n\n"
                f"Reference Token: DOC_REF_{uuid.uuid4().hex[:8].upper()}."
            )

            records.append({
                "document_id": f"doc_{i+1:05d}",
                "content": full_content,
                "source": f"corpus_stream_{i % 3 + 1}",
                "category": cat,
                "timestamp": f"2026-08-20T{(i % 24):02d}:{(i % 60):02d}:00Z"
            })

        return pd.DataFrame(records)

    def generate_dense_embedding(self, text: str) -> np.ndarray:
        """Synthesizes a realistic high-dimensional semantic vector from text content."""
        words = text.lower().split()
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        
        for w in words:
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest()[:8], 16)
            idx = h % self.embedding_dim
            sign = 1.0 if (h >> 3) & 1 else -1.0
            vec[idx] += sign

        # Semantic cluster offset
        first_word = words[0] if words else "general"
        cluster_seed = int(hashlib.md5(first_word.encode("utf-8")).hexdigest()[:4], 16)
        rng = np.random.RandomState(cluster_seed)
        noise = rng.normal(0, 0.05, self.embedding_dim).astype(np.float32)
        vec += noise

        # L2 normalize
        return l2_normalize(vec)

    def run_simulation_pipeline(
        self,
        raw_df: pd.DataFrame,
        quantization_mode: str = "INT8"
    ) -> Tuple[pd.DataFrame, PipelineAuditReport, Dict[str, Any]]:
        """Executes full end-to-end simulated distributed pipeline with realistic stage telemetry."""
        job_id = f"sim_{uuid.uuid4().hex[:10]}"
        started_at = "2026-08-20T11:00:00Z"
        start_clock = time.time()
        stage_metrics: List[BatchStageMetrics] = []

        # Stage 1: Ingestion
        t0 = time.time()
        input_count = len(raw_df)
        d1 = max(time.time() - t0, 0.005)
        stage_metrics.append(BatchStageMetrics(
            stage_name="IngestStage",
            input_records=input_count,
            output_records=input_count,
            duration_seconds=round(d1, 3),
            throughput_records_per_sec=round(input_count / d1, 2)
        ))

        # Stage 2: Clean & Deduplicate
        t0 = time.time()
        cleaned_records = []
        seen_hashes = set()
        for _, row in raw_df.iterrows():
            cleaned = self.normalizer.clean(str(row["content"]))
            chash = ExactHasher.compute_sha256(cleaned)
            if chash not in seen_hashes:
                seen_hashes.add(chash)
                cleaned_records.append({
                    "document_id": row["document_id"],
                    "cleaned_content": cleaned,
                    "content_hash": chash,
                    "source": row["source"],
                    "category": row["category"]
                })
        cleaned_df = pd.DataFrame(cleaned_records)
        d2 = max(time.time() - t0, 0.01)
        stage_metrics.append(BatchStageMetrics(
            stage_name="CleaningAndDedupStage",
            input_records=input_count,
            output_records=len(cleaned_df),
            duration_seconds=round(d2, 3),
            throughput_records_per_sec=round(len(cleaned_df) / d2, 2)
        ))

        # Stage 3: Chunking & Token Extraction
        t0 = time.time()
        chunk_rows = []
        for _, row in cleaned_df.iterrows():
            chunks = self.chunker.chunk_text(
                document_id=row["document_id"],
                text=row["cleaned_content"],
                source=row["source"],
                category=row["category"]
            )
            for c in chunks:
                chunk_rows.append({
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "chunk_index": c.chunk_index,
                    "chunk_content": c.chunk_content,
                    "token_count": c.token_count,
                    "source": c.source,
                    "category": c.category
                })
        chunk_df = pd.DataFrame(chunk_rows)
        d3 = max(time.time() - t0, 0.015)
        stage_metrics.append(BatchStageMetrics(
            stage_name="ChunkingExplodeStage",
            input_records=len(cleaned_df),
            output_records=len(chunk_df),
            duration_seconds=round(d3, 3),
            throughput_records_per_sec=round(len(chunk_df) / d3, 2)
        ))

        # Stage 4: Embedding Generation & L2 Normalization
        t0 = time.time()
        vectors_list: List[np.ndarray] = []
        for _, row in chunk_df.iterrows():
            vec = self.generate_dense_embedding(row["chunk_content"])
            vectors_list.append(vec)
        vector_matrix = np.vstack(vectors_list) if vectors_list else np.zeros((0, self.embedding_dim), dtype=np.float32)
        d4 = max(time.time() - t0, 0.02)
        stage_metrics.append(BatchStageMetrics(
            stage_name="VectorEmbeddingStage",
            input_records=len(chunk_df),
            output_records=len(chunk_df),
            duration_seconds=round(d4, 3),
            throughput_records_per_sec=round(len(chunk_df) / d4, 2)
        ))

        # Stage 5: INT8 Quantization & Partition Sharding
        t0 = time.time()
        q_vectors, scales, zero_points = quantize_int8(vector_matrix, self.clip_threshold)
        
        # Calculate quantization metrics
        deq_matrix = dequantize_int8(q_vectors, scales, zero_points)
        quant_error = compute_quantization_error(vector_matrix, deq_matrix)

        # Build output dataframe
        final_rows = []
        for idx, row in chunk_df.iterrows():
            part_id = int(hashlib.md5(row["chunk_id"].encode("utf-8")).hexdigest()[:4], 16) % self.num_partitions
            final_rows.append({
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "chunk_index": row["chunk_index"],
                "chunk_content": row["chunk_content"],
                "category": row["category"],
                "source": row["source"],
                "vector_fp32": vector_matrix[idx].tolist(),
                "vector_int8_bytes": q_vectors[idx].tobytes(),
                "scale_factor": float(scales[idx]),
                "partition_id": part_id
            })
        final_df = pd.DataFrame(final_rows)
        d5 = max(time.time() - t0, 0.01)
        stage_metrics.append(BatchStageMetrics(
            stage_name="QuantizationAndPartitionStage",
            input_records=len(chunk_df),
            output_records=len(final_df),
            duration_seconds=round(d5, 3),
            throughput_records_per_sec=round(len(final_df) / d5, 2)
        ))

        total_duration = max(time.time() - start_clock, 0.05)
        audit_report = PipelineAuditReport(
            job_id=job_id,
            status="SUCCESS",
            started_at=started_at,
            completed_at="2026-08-20T11:00:05Z",
            total_duration_seconds=round(total_duration, 3),
            total_input_records=input_count,
            total_output_vectors=len(final_df),
            overall_throughput_records_per_sec=round(len(final_df) / total_duration, 2),
            stage_metrics=stage_metrics,
            quantization_mode=quantization_mode,
            vector_dimension=self.embedding_dim,
            num_partitions=self.num_partitions,
            storage_sink="parquet"
        )

        analytics = {
            "vector_matrix": vector_matrix,
            "quantized_matrix": q_vectors,
            "quantization_error": quant_error,
            "partition_distribution": final_df["partition_id"].value_counts().to_dict()
        }

        return final_df, audit_report, analytics
