"""Core feature processing, vector mathematical operations, and schema definitions."""

from src.core.chunker import TextChunker
from src.core.deduplicator import ExactHasher, MinHashLSH, SimHasher
from src.core.normalizer import TextNormalizer
from src.core.schemas import (
    BatchStageMetrics,
    CleanedDocument,
    DocumentChunk,
    PipelineAuditReport,
    RawDocument,
    VectorPayload,
)
from src.core.vector_ops import (
    compute_cosine_matrix,
    compute_quantization_error,
    dequantize_int8,
    l2_normalize,
    project_pca,
    quantize_int8,
)

__all__ = [
    "RawDocument",
    "CleanedDocument",
    "DocumentChunk",
    "VectorPayload",
    "BatchStageMetrics",
    "PipelineAuditReport",
    "TextNormalizer",
    "TextChunker",
    "ExactHasher",
    "SimHasher",
    "MinHashLSH",
    "l2_normalize",
    "quantize_int8",
    "dequantize_int8",
    "compute_quantization_error",
    "compute_cosine_matrix",
    "project_pca",
]
