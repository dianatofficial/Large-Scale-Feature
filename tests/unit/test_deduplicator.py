from src.core.deduplicator import ExactHasher, SimHasher, MinHashLSH

def test_exact_hash_deterministic():
    text1 = "Deterministic distributed vector processing engine"
    text2 = "Deterministic distributed vector processing engine"
    text3 = "Different content payload"
    
    h1 = ExactHasher.compute_sha256(text1)
    h2 = ExactHasher.compute_sha256(text2)
    h3 = ExactHasher.compute_sha256(text3)
    
    assert h1 == h2
    assert h1 != h3

def test_simhash_near_duplicates():
    simhasher = SimHasher()
    tokens1 = ["distributed", "pyspark", "arrow", "feature", "pipeline", "execution"]
    tokens2 = ["distributed", "pyspark", "arrow", "feature", "pipeline", "execution", "tuning"]
    tokens3 = ["genomics", "crispr", "sequencing", "transcriptional", "biology"]

    fp1 = simhasher.compute_fingerprint(tokens1)
    fp2 = simhasher.compute_fingerprint(tokens2)
    fp3 = simhasher.compute_fingerprint(tokens3)

    dist_near = simhasher.hamming_distance(fp1, fp2)
    dist_far = simhasher.hamming_distance(fp1, fp3)

    assert dist_near < dist_far
    assert dist_near <= 12

def test_minhash_jaccard_estimation():
    lsh = MinHashLSH(num_perm=64, seed=42)
    t1 = ["apple", "banana", "cherry", "date", "fig"]
    t2 = ["apple", "banana", "cherry", "date", "grape"]
    t3 = ["quantum", "physics", "relativity", "spacetime"]

    sig1 = lsh.compute_signature(t1)
    sig2 = lsh.compute_signature(t2)
    sig3 = lsh.compute_signature(t3)

    sim_high = lsh.estimate_jaccard(sig1, sig2)
    sim_low = lsh.estimate_jaccard(sig1, sig3)

    assert sim_high > sim_low
    assert sim_high >= 0.5
