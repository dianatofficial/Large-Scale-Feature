from typing import Dict, Optional, Tuple
import numpy as np

def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Applies vectorized L2 Unit Normalization (v / max(||v||_2, eps))."""
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        norm = max(float(norm), eps)
        return (vectors / norm).astype(np.float32)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (vectors / norms).astype(np.float32)

def quantize_int8(
    vectors: np.ndarray, 
    clip_threshold: float = 2.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized Symmetric INT8 Scalar Quantization with per-vector dynamic scaling.
    Formulation:
      max_val = max(|x|)
      scale = max_val / 127.0
      q = round(x / scale) in [-128, 127]
    Returns:
      quantized_array (np.int8), scale_factors (np.float32), zero_points (np.int8)
    """
    is_1d = vectors.ndim == 1
    matrix = np.expand_dims(vectors, axis=0) if is_1d else vectors

    # Per-vector dynamic scale calculation with fallback epsilon
    max_vals = np.max(np.abs(matrix), axis=1)
    max_vals = np.maximum(max_vals, 1e-7)
    
    # Optional threshold ceiling
    if clip_threshold > 0:
        max_vals = np.minimum(max_vals, clip_threshold)

    scales = (max_vals / 127.0).astype(np.float32)
    zero_points = np.zeros((matrix.shape[0],), dtype=np.int8)

    scaled = matrix / scales[:, np.newaxis]
    quantized = np.clip(np.round(scaled), -128, 127).astype(np.int8)

    if is_1d:
        return quantized[0], scales, zero_points
    return quantized, scales, zero_points

def dequantize_int8(
    quantized: np.ndarray, 
    scales: np.ndarray, 
    zero_points: Optional[np.ndarray] = None
) -> np.ndarray:
    """Reconstructs approximate FP32 vectors from INT8 quantized representations."""
    is_1d = quantized.ndim == 1
    q_mat = np.expand_dims(quantized, axis=0) if is_1d else quantized

    if scales.ndim == 0 or (scales.ndim == 1 and scales.shape[0] == 1 and q_mat.shape[0] > 1):
        s_mat = np.full((q_mat.shape[0], 1), float(scales), dtype=np.float32)
    else:
        s_mat = scales[:, np.newaxis] if scales.ndim == 1 else scales

    reconstructed = (q_mat.astype(np.float32) * s_mat).astype(np.float32)
    return reconstructed[0] if is_1d else reconstructed

def compute_quantization_error(original: np.ndarray, dequantized: np.ndarray) -> Dict[str, float]:
    """Calculates Mean Squared Error (MSE), Cosine Distortion, and Signal-to-Noise Ratio."""
    diff = original - dequantized
    mse = float(np.mean(diff ** 2))
    max_err = float(np.max(np.abs(diff)))

    # Cosine Similarity between original and reconstructed
    norm_orig = np.linalg.norm(original)
    norm_deq = np.linalg.norm(dequantized)
    if norm_orig > 1e-12 and norm_deq > 1e-12:
        cosine_sim = float(np.dot(original.flatten(), dequantized.flatten()) / (norm_orig * norm_deq))
    else:
        cosine_sim = 1.0

    cosine_distortion = max(0.0, 1.0 - cosine_sim)
    snr_db = 10.0 * np.log10(np.var(original) / (mse + 1e-12)) if mse > 0 else 100.0

    return {
        "mse": mse,
        "max_absolute_error": max_err,
        "cosine_similarity": cosine_sim,
        "cosine_distortion": cosine_distortion,
        "snr_db": float(snr_db)
    }

def compute_cosine_matrix(matrix_a: np.ndarray, matrix_b: Optional[np.ndarray] = None) -> np.ndarray:
    """Computes full pairwise cosine similarity matrix between L2 normalized arrays."""
    norm_a = l2_normalize(matrix_a)
    if matrix_b is None:
        return np.dot(norm_a, norm_a.T)
    norm_b = l2_normalize(matrix_b)
    return np.dot(norm_a, norm_b.T)

def project_pca(matrix: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Fast Truncated SVD / PCA projection for high-dimensional vector space visualization."""
    if matrix.shape[0] < n_components:
        proj = np.zeros((matrix.shape[0], n_components), dtype=np.float32)
        proj[:, :matrix.shape[1]] = matrix[:, :min(matrix.shape[1], n_components)]
        return proj, np.ones(n_components, dtype=np.float32) / n_components

    mean = np.mean(matrix, axis=0)
    centered = matrix - mean

    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    projected = np.dot(centered, components.T)

    total_var = np.sum(s ** 2)
    explained_var = (s[:n_components] ** 2) / (total_var + 1e-12)

    return projected.astype(np.float32), explained_var.astype(np.float32)
