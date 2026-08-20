"""Core feature processing, vector mathematical operations, and schema definitions."""
from src.core.schemas import RawDocument, CleanedDocument, DocumentChunk, VectorPayload, BatchStageMetrics, PipelineAuditReport
from src.core.normalizer import TextNormalizer
from src.core.chunker import TextChunker
from src.core.deduplicator import ExactHasher, SimHasher, MinHashLSH
from src.core.vector_ops import (
    l2_normalize,
    quantize_int8,
    dequantize_int8,
    compute_quantization_error,
    compute_cosine_matrix,
    project_pca
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
    "project_pca"
]
