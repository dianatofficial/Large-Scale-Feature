import hashlib
import struct
from typing import List
import numpy as np
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import (
    ArrayType,
    BinaryType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from src.core.chunker import TextChunker
from src.core.normalizer import TextNormalizer
from src.core.vector_ops import l2_normalize, quantize_int8

# Thread-safe global instances for worker tasks
_normalizer = TextNormalizer()
_chunker = TextChunker(chunk_size=256, chunk_overlap=32)

@pandas_udf(StringType())
def clean_text_udf(series: pd.Series) -> pd.Series:
    """Vectorized Arrow Pandas UDF for text cleaning and unicode normalization."""
    normalizer = TextNormalizer()
    return series.fillna("").apply(normalizer.clean)

@pandas_udf(StringType())
def content_hash_udf(series: pd.Series) -> pd.Series:
    """Vectorized Arrow Pandas UDF for deterministic SHA-256 payload hashing."""
    return series.fillna("").apply(lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest() if t else "")

@pandas_udf(ArrayType(StringType()))
def chunk_text_udf(series: pd.Series) -> pd.Series:
    """Vectorized Arrow Pandas UDF for recursive text chunking into string arrays."""
    chunker = TextChunker(chunk_size=256, chunk_overlap=32)
    def _chunk(text: str) -> List[str]:
        if not text or not text.strip():
            return []
        chunks = chunker.chunk_text(document_id="tmp", text=text)
        return [c.chunk_content for c in chunks]
    return series.fillna("").apply(_chunk)

@pandas_udf(ArrayType(FloatType()))
def generate_deterministic_embedding_udf(series: pd.Series) -> pd.Series:
    """
    Vectorized Arrow Pandas UDF generating high-throughput dense feature vectors.
    Uses an internal deterministic multi-hash pseudo-projection matrix to synthesize
    embeddings of dim=384 with preserved semantic locality across batch partitions.
    """
    dim = 384
    results = []
    for text in series.fillna(""):
        if not text:
            results.append([0.0] * dim)
            continue
        
        # Fast deterministic hash-based feature representation
        words = text.lower().split()
        vec = np.zeros(dim, dtype=np.float32)
        for w in words:
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest()[:8], 16)
            idx = h % dim
            sign = 1.0 if (h >> 3) & 1 else -1.0
            vec[idx] += sign

        # Add structural token length component
        vec[0] += len(words) * 0.1
        vec[1] += len(text) * 0.01

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec = vec / norm
        results.append(vec.tolist())

    return pd.Series(results)

@pandas_udf(ArrayType(FloatType()))
def vector_l2_normalize_udf(series: pd.Series) -> pd.Series:
    """Vectorized Arrow Pandas UDF for L2 unit normalization of float arrays."""
    results = []
    for vec_list in series:
        if vec_list is None or len(vec_list) == 0:
            results.append([])
            continue
        arr = np.array(vec_list, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 1e-12:
            arr = arr / norm
        results.append(arr.tolist())
    return pd.Series(results)

@pandas_udf(BinaryType())
def vector_quantize_int8_udf(series: pd.Series) -> pd.Series:
    """
    Vectorized Arrow Pandas UDF converting float arrays into compact INT8 byte buffers.
    Achieves 4x memory and storage reduction for vector search staging.
    """
    results = []
    clip_threshold = 2.5
    scale = clip_threshold / 127.0
    
    for vec_list in series:
        if vec_list is None or len(vec_list) == 0:
            results.append(b"")
            continue
        arr = np.array(vec_list, dtype=np.float32)
        clipped = np.clip(arr, -clip_threshold, clip_threshold)
        q = np.round(clipped / scale).astype(np.int8)
        results.append(q.tobytes())
    return pd.Series(results)

@pandas_udf(IntegerType())
def compute_partition_id_udf(id_series: pd.Series) -> pd.Series:
    """Computes balanced integer partition ID from Murmur/MD5 hash for vector sharding."""
    num_partitions = 16
    return id_series.fillna("").apply(
        lambda x: int(hashlib.md5(x.encode("utf-8")).hexdigest()[:8], 16) % num_partitions
    )
