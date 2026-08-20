import numpy as np
from src.core.vector_ops import (
    l2_normalize,
    quantize_int8,
    dequantize_int8,
    compute_quantization_error,
    compute_cosine_matrix,
    project_pca
)

def test_l2_normalization_1d():
    v = np.array([3.0, 4.0], dtype=np.float32)
    norm_v = l2_normalize(v)
    assert np.allclose(norm_v, np.array([0.6, 0.8], dtype=np.float32))
    assert np.isclose(np.linalg.norm(norm_v), 1.0)

def test_l2_normalization_2d(sample_vector_matrix):
    norm_mat = l2_normalize(sample_vector_matrix)
    row_norms = np.linalg.norm(norm_mat, axis=1)
    assert np.allclose(row_norms, 1.0, atol=1e-5)

def test_int8_quantization_fidelity(sample_vector_matrix):
    norm_mat = l2_normalize(sample_vector_matrix)
    q_vecs, scales, zps = quantize_int8(norm_mat, clip_threshold=2.5)

    assert q_vecs.dtype == np.int8
    assert np.all(q_vecs >= -128) and np.all(q_vecs <= 127)

    deq_mat = dequantize_int8(q_vecs, scales, zps)
    err = compute_quantization_error(norm_mat, deq_mat)

    # Cosine fidelity should exceed 99.0%
    assert err["cosine_similarity"] > 0.990
    assert err["mse"] < 1e-3
    assert err["snr_db"] > 25.0

def test_cosine_matrix(sample_vector_matrix):
    mat = sample_vector_matrix[:5]
    cos_mat = compute_cosine_matrix(mat)
    
    assert cos_mat.shape == (5, 5)
    # Diagonal of self-similarity must equal 1.0
    assert np.allclose(np.diag(cos_mat), 1.0, atol=1e-5)

def test_pca_projection(sample_vector_matrix):
    proj, exp_var = project_pca(sample_vector_matrix, n_components=3)
    assert proj.shape == (20, 3)
    assert len(exp_var) == 3
    assert np.sum(exp_var) <= 1.0001
