import hashlib
from typing import List, Set

class ExactHasher:
    """Fast deterministic cryptographic hashing for exact document deduplication."""

    @staticmethod
    def compute_sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def compute_md5(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

class SimHasher:
    """64-bit SimHash fingerprinting for near-duplicate text detection."""

    def __init__(self, bit_length: int = 64):
        self.bit_length = bit_length

    def compute_fingerprint(self, tokens: List[str]) -> int:
        """Computes 64-bit integer SimHash fingerprint from token list."""
        if not tokens:
            return 0

        v = [0] * self.bit_length
        for token in tokens:
            token_hash = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:16], 16)
            for i in range(self.bit_length):
                bitmask = 1 << i
                if token_hash & bitmask:
                    v[i] += 1
                else:
                    v[i] -= 1

        fingerprint = 0
        for i in range(self.bit_length):
            if v[i] > 0:
                fingerprint |= (1 << i)

        return fingerprint

    @staticmethod
    def hamming_distance(hash1: int, hash2: int) -> int:
        """Computes bit-level Hamming distance between two fingerprints."""
        x = hash1 ^ hash2
        distance = 0
        while x:
            distance += 1
            x &= x - 1
        return distance

class MinHashLSH:
    """MinHash signature generation for fast Jaccard similarity estimation."""

    def __init__(self, num_perm: int = 64, seed: int = 42):
        self.num_perm = num_perm
        self.seed = seed
        self.prime = 4294967311  # Large 32-bit prime
        
        # Deterministic linear permutation parameters: h_i(x) = (a_i * x + b_i) % prime
        import random
        rng = random.Random(seed)
        self.a_params = [rng.randint(1, self.prime - 1) for _ in range(num_perm)]
        self.b_params = [rng.randint(0, self.prime - 1) for _ in range(num_perm)]

    def compute_signature(self, tokens: List[str]) -> List[int]:
        """Computes k-dimensional MinHash signature vector."""
        if not tokens:
            return [0] * self.num_perm

        token_hashes = [int(hashlib.md5(t.encode("utf-8")).hexdigest()[:8], 16) for t in set(tokens)]
        signature = [float("inf")] * self.num_perm

        for th in token_hashes:
            for i in range(self.num_perm):
                permuted = (self.a_params[i] * th + self.b_params[i]) % self.prime
                if permuted < signature[i]:
                    signature[i] = permuted

        return [int(s) if s != float("inf") else 0 for s in signature]

    @staticmethod
    def estimate_jaccard(sig1: List[int], sig2: List[int]) -> float:
        """Estimates Jaccard similarity from MinHash signatures."""
        if len(sig1) != len(sig2) or len(sig1) == 0:
            return 0.0
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return float(matches) / len(sig1)
