"""
core/features/kmers.py
----------------------
K-mer frequency feature extractor for Project NYX.

Converts a Sequence object into a normalized numpy feature vector
suitable for sklearn estimators.

Design decisions:
    - k=1, k=2: Full alphabet vectors. Dense, interpretable, fast.
      Protein: 20^k features. DNA: 4^k features.
    - k=3, k=4: Hashed vectors capped at `max_features` (default 512).
      High-k full-alphabet vectors (8k, 160k) are too sparse for short
      sequences and slow to compute. HashingVectorizer trades
      interpretability for practicality. A WARNING is logged every time
      k >= 3 is used — this is intentional so you never forget the tradeoff.
    - Normalization: frequency (count / total_kmers), not raw count.
      Sequences of different lengths become comparable.
    - Sequence type: explicit ('protein' | 'dna') is preferred.
      Auto-detect is provided as fallback but logs a warning.
"""

import logging
from collections import Counter
from itertools import product
from typing import Literal, Optional

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DNA_ALPHABET: list[str] = ["A", "T", "G", "C"]
PROTEIN_ALPHABET: list[str] = list("ACDEFGHIKLMNPQRSTVWY")

SeqType = Literal["protein", "dna"]

# Threshold: if >85% of unique chars are in DNA alphabet, call it DNA.
_DNA_CHAR_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class KmerExtractor:
    """
    Extracts k-mer frequency features from biological sequences.

    Args:
        k_values:     List of k values to compute. Each k produces its own
                      feature block; all blocks are concatenated in the
                      returned vector. Default: [1, 2].
        seq_type:     'protein' or 'dna'. When None, auto-detection is used
                      (not recommended — be explicit).
        max_features: Feature cap applied only when k >= 3. Has no effect
                      on k=1 or k=2 (those use full alphabets).
                      Default: 512.

    Example:
        extractor = KmerExtractor(k_values=[1, 2], seq_type="protein")
        vector = extractor.extract(sequence_obj)
        # vector.shape -> (420,)  [20 (k=1) + 400 (k=2)]
    """

    def __init__(
        self,
        k_values: list[int] = None,
        seq_type: Optional[SeqType] = None,
        max_features: int = 512,
    ) -> None:
        self.k_values: list[int] = k_values if k_values is not None else [1, 2]
        self.seq_type: Optional[SeqType] = seq_type
        self.max_features: int = max_features

        self._validate_init()
        logger.debug(
            "KmerExtractor init: k_values=%s, seq_type=%s, max_features=%d",
            self.k_values,
            self.seq_type or "auto-detect",
            self.max_features,
        )

    def extract(self, sequence) -> np.ndarray:
        """
        Extract a normalized k-mer frequency vector from a Sequence object.

        Args:
            sequence: A Sequence object (from core.sequence.loader) with
                      a .sequence attribute (str).

        Returns:
            1D numpy array of float32. Shape: (n_features,) where n_features
            is the sum of feature counts across all requested k values.

        Raises:
            ValueError: If sequence.sequence is empty or shorter than min(k_values).
        """
        raw: str = sequence.sequence.upper().strip()
        self._validate_sequence(raw)

        resolved_type = self._resolve_seq_type(raw)
        alphabet = PROTEIN_ALPHABET if resolved_type == "protein" else DNA_ALPHABET

        blocks: list[np.ndarray] = []
        for k in self.k_values:
            block = self._extract_one_k(raw, k, alphabet)
            blocks.append(block)
            logger.debug(
                "id=%s k=%d block_shape=%s", sequence.id, k, block.shape
            )

        vector = np.concatenate(blocks).astype(np.float32)
        logger.debug("id=%s final_vector_shape=%s", sequence.id, vector.shape)
        return vector

    def feature_count(self, seq_type: Optional[SeqType] = None) -> int:
        """
        Return the total number of features this extractor will produce.

        Useful for pre-allocating feature matrices.

        Args:
            seq_type: Override the instance seq_type for this calculation.
                      Falls back to instance setting.

        Returns:
            Integer count of features.
        """
        resolved = seq_type or self.seq_type
        if resolved is None:
            raise ValueError(
                "Cannot compute feature_count without a known seq_type. "
                "Pass seq_type='protein' or seq_type='dna'."
            )
        alphabet_size = len(PROTEIN_ALPHABET) if resolved == "protein" else len(DNA_ALPHABET)
        total = 0
        for k in self.k_values:
            if k <= 2:
                total += alphabet_size ** k
            else:
                total += self.max_features
        return total

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_init(self) -> None:
        """Fail fast on bad constructor arguments."""
        if not self.k_values:
            raise ValueError("k_values must be a non-empty list.")
        for k in self.k_values:
            if k < 1 or k > 4:
                raise ValueError(f"k must be between 1 and 4. Got: {k}")
        if self.max_features < 1:
            raise ValueError(f"max_features must be >= 1. Got: {self.max_features}")
        if self.seq_type not in (None, "protein", "dna"):
            raise ValueError(f"seq_type must be 'protein', 'dna', or None. Got: {self.seq_type!r}")

    def _validate_sequence(self, raw: str) -> None:
        """
        Ensure the sequence is usable.

        Args:
            raw: Uppercased, stripped sequence string.

        Raises:
            ValueError: If empty or shorter than the smallest k requested.
        """
        if not raw:
            raise ValueError("Cannot extract features from an empty sequence.")
        min_k = min(self.k_values)
        if len(raw) < min_k:
            raise ValueError(
                f"Sequence length {len(raw)} is shorter than min k={min_k}. "
                "Cannot extract any k-mers."
            )

    def _resolve_seq_type(self, raw: str) -> SeqType:
        """
        Return the sequence type to use for alphabet selection.

        Prefers the explicit instance setting. Falls back to auto-detection
        and logs a warning so the caller knows it's happening.

        Args:
            raw: Uppercased sequence string.

        Returns:
            'protein' or 'dna'.
        """
        if self.seq_type is not None:
            return self.seq_type

        detected = _autodetect_seq_type(raw)
        logger.warning(
            "seq_type not specified — auto-detected as '%s'. "
            "Pass seq_type explicitly to KmerExtractor for reliable results.",
            detected,
        )
        return detected

    def _extract_one_k(
        self, raw: str, k: int, alphabet: list[str]
    ) -> np.ndarray:
        """
        Compute normalized k-mer frequency vector for a single k.

        k=1 or k=2: full-alphabet dense vector.
        k>=3: hashed vector capped at self.max_features.

        Args:
            raw:      Uppercased sequence string.
            k:        K-mer length.
            alphabet: Character list for the sequence type.

        Returns:
            1D numpy float32 array.
        """
        if k >= 3:
            logger.warning(
                "k=%d requested. Using HashingVectorizer with max_features=%d. "
                "Features are NOT interpretable at this k. "
                "Use k=1,2 for production models until you have enough data.",
                k,
                self.max_features,
            )
            return _hashed_kmer_vector(raw, k, self.max_features)

        return _full_alphabet_kmer_vector(raw, k, alphabet)


# ---------------------------------------------------------------------------
# Module-level functions (stateless, easy to unit-test)
# ---------------------------------------------------------------------------


def _full_alphabet_kmer_vector(
    sequence: str, k: int, alphabet: list[str]
) -> np.ndarray:
    """
    Build a normalized frequency vector over all possible k-mers for
    the given alphabet.

    Only k-mers composed entirely of alphabet characters are counted;
    ambiguous chars (N, X, B, Z, ...) are silently ignored.

    Args:
        sequence: Uppercased sequence string.
        k:        K-mer length (1 or 2 recommended).
        alphabet: List of valid characters.

    Returns:
        1D float32 array of shape (len(alphabet)^k,).
        Zero vector if no valid k-mers are found.
    """
    alphabet_set = set(alphabet)

    # Build ordered index: kmer string -> position in vector
    all_kmers: list[str] = ["".join(p) for p in product(alphabet, repeat=k)]
    kmer_index: dict[str, int] = {km: i for i, km in enumerate(all_kmers)}

    counts = Counter(
        sequence[i : i + k]
        for i in range(len(sequence) - k + 1)
        if all(c in alphabet_set for c in sequence[i : i + k])
    )

    vector = np.zeros(len(all_kmers), dtype=np.float32)
    total = sum(counts.values())

    if total == 0:
        logger.warning(
            "No valid k=%d mers found in sequence (all chars may be ambiguous).", k
        )
        return vector

    for kmer, count in counts.items():
        if kmer in kmer_index:
            vector[kmer_index[kmer]] = count / total

    return vector


def _hashed_kmer_vector(sequence: str, k: int, max_features: int) -> np.ndarray:
    """
    Approximate k-mer frequency vector using sklearn's HashingVectorizer.

    Used for k>=3 where full alphabets produce impractically large vectors.
    Output dimensions are fixed at max_features regardless of actual k-mer
    diversity. Features are NOT human-interpretable (hash collision).

    Args:
        sequence:     Uppercased sequence string.
        k:            K-mer length.
        max_features: Output vector length.

    Returns:
        1D float32 array of shape (max_features,).
        Values are L1-normalized by default from HashingVectorizer.
    """
    kmers_as_tokens = " ".join(
        sequence[i : i + k] for i in range(len(sequence) - k + 1)
    )

    if not kmers_as_tokens.strip():
        return np.zeros(max_features, dtype=np.float32)

    vectorizer = HashingVectorizer(
        analyzer="word",
        n_features=max_features,
        norm="l1",       # gives frequency-like values (sum=1)
        alternate_sign=False,  # all values positive
    )
    sparse = vectorizer.fit_transform([kmers_as_tokens])
    return sparse.toarray()[0].astype(np.float32)


def _autodetect_seq_type(sequence: str) -> SeqType:
    """
    Heuristic: classify a sequence as DNA or protein.

    Logic: if >= 85% of unique characters in the sequence are in the
    DNA alphabet {A, T, G, C}, call it DNA. Otherwise protein.
    This handles sequences with N (ambiguous nucleotide) conservatively.

    Args:
        sequence: Uppercased sequence string.

    Returns:
        'dna' or 'protein'.
    """
    unique_chars = set(sequence.upper())
    dna_chars = set(DNA_ALPHABET)
    overlap = len(unique_chars & dna_chars)
    ratio = overlap / len(unique_chars) if unique_chars else 0.0
    return "dna" if ratio >= _DNA_CHAR_THRESHOLD else "protein"


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
#
#   python core/features/kmers.py
#
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")  # allow running from nyx/ root

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    # -- Minimal stub so we don't depend on loader.py for this smoke test --
    from dataclasses import dataclass, field as dc_field

    @dataclass
    class _StubSeq:
        id: str
        sequence: str
        description: str = ""
        metadata: dict = dc_field(default_factory=dict)

    # --- Protein sequence (enzyme fragment) ---
    protein_seq = _StubSeq(
        id="prot_001",
        sequence="MKTIIALSYIFCLVFA" * 5,  # 80 aa, repeated motif
    )

    ext_protein = KmerExtractor(k_values=[1, 2], seq_type="protein")
    vec = ext_protein.extract(protein_seq)
    print(f"\n[Protein k=1,2] shape={vec.shape}  expected=(420,)")
    print(f"  Non-zero features: {np.count_nonzero(vec)}")
    print(f"  Vector sum (should be ~2.0 for two k blocks): {vec.sum():.4f}")

    # --- DNA sequence ---
    dna_seq = _StubSeq(
        id="dna_001",
        sequence="ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
    )

    ext_dna = KmerExtractor(k_values=[1, 2], seq_type="dna")
    vec_dna = ext_dna.extract(dna_seq)
    print(f"\n[DNA k=1,2] shape={vec_dna.shape}  expected=(20,)")
    print(f"  Non-zero features: {np.count_nonzero(vec_dna)}")

    # --- k=3 protein (hashed, triggers WARNING) ---
    ext_k3 = KmerExtractor(k_values=[3], seq_type="protein", max_features=512)
    vec_k3 = ext_k3.extract(protein_seq)
    print(f"\n[Protein k=3 hashed] shape={vec_k3.shape}  expected=(512,)")
    print(f"  Non-zero features: {np.count_nonzero(vec_k3)}")

    # --- feature_count helper ---
    print(f"\nfeature_count (protein, k=[1,2]): {ext_protein.feature_count()}")
    print(f"feature_count (dna,     k=[1,2]): {ext_dna.feature_count()}")

    # --- Auto-detect smoke test ---
    ext_auto = KmerExtractor(k_values=[1], seq_type=None)
    vec_auto = ext_auto.extract(dna_seq)
    print(f"\n[Auto-detect DNA] shape={vec_auto.shape}  (triggers WARNING above)")
