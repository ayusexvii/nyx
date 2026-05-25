"""
core/sequence/loader.py
-----------------------
Sequence loader for Project NYX.
Handles FASTA (plain and gzipped) with EC number extraction from headers.
Future: FASTQ, GenBank support via the same SequenceLoader interface.
"""

import gzip
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

logger = logging.getLogger(__name__)

# Matches both "EC:1.2.3.4" and "ec_number=1.2.3.4"
_EC_PATTERNS = [
    re.compile(r"EC:(\d+\.\d+\.\d+\.\d+)", re.IGNORECASE),
    re.compile(r"ec_number=(\d+\.\d+\.\d+\.\d+)", re.IGNORECASE),
]


@dataclass
class Sequence:
    """
    Represents a single biological sequence with parsed metadata.

    Attributes:
        id:          Sequence identifier (everything before first space in header).
        sequence:    Raw sequence string (DNA, RNA, or protein).
        description: Full header line from FASTA (including id).
        metadata:    Arbitrary key-value store for extracted annotations
                     (e.g. ec_number, source_file, original_length).
    """

    id: str
    sequence: str
    description: str
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.sequence)

    def __repr__(self) -> str:
        ec = self.metadata.get("ec_number", "none")
        return f"Sequence(id={self.id!r}, len={len(self)}, ec={ec!r})"


class SequenceLoader:
    """
    Loads biological sequences from FASTA files (plain or gzipped).

    Usage:
        loader = SequenceLoader()
        sequences = loader.load("path/to/seqs.fasta")
        sequences = loader.load("path/to/seqs.fasta.gz")

    Design notes:
        - Returns List[Sequence], not a generator, so callers can index freely.
        - Malformed records are logged and skipped — pipeline never crashes on
          one bad sequence.
        - EC numbers are extracted from headers and stored in metadata.
        - Source file path is always stored in metadata["source_file"].
    """

    SUPPORTED_FORMATS = {
        ".fasta": "fasta",
        ".fa": "fasta",
        ".fna": "fasta",
        ".faa": "fasta",
        ".fastq": "fastq",
        ".fq": "fastq",
    }

    def load(self, filepath: str | Path) -> list[Sequence]:
        """
        Load sequences from a FASTA (or FASTQ) file.

        Args:
            filepath: Path to the sequence file. Accepts plain or .gz compressed.

        Returns:
            List of Sequence objects. Empty list if file has no valid records.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError:        If the file extension is not supported.
        """
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Sequence file not found: {path}")

        fmt = self._detect_format(path)
        logger.info("Loading sequences from %s (format=%s)", path.name, fmt)

        records = self._parse(path, fmt)
        sequences = self._convert(records, source_file=str(path))

        logger.info("Loaded %d sequences from %s", len(sequences), path.name)
        return sequences

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_format(self, path: Path) -> str:
        """
        Resolve Bio format string from file extension, stripping .gz first.

        Args:
            path: File path, may end in .gz.

        Returns:
            Biopython format string, e.g. "fasta".

        Raises:
            ValueError: If the base extension is not in SUPPORTED_FORMATS.
        """
        name = path.name
        if name.endswith(".gz"):
            name = name[:-3]  # strip .gz, then check base extension
        ext = Path(name).suffix.lower()

        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file extension '{ext}'. "
                f"Supported: {list(self.SUPPORTED_FORMATS)}"
            )
        return self.SUPPORTED_FORMATS[ext]

    def _parse(self, path: Path, fmt: str) -> list[SeqRecord]:
        """
        Open plain or gzipped file and parse with Biopython.

        Args:
            path: Validated file path.
            fmt:  Biopython format string.

        Returns:
            List of SeqRecord objects (may be empty).
        """
        opener = gzip.open if path.name.endswith(".gz") else open
        mode = "rt"  # text mode for both openers

        try:
            with opener(path, mode) as handle:
                return list(SeqIO.parse(handle, fmt))
        except Exception as exc:
            logger.error("Failed to parse %s: %s", path.name, exc)
            raise

    def _convert(self, records: list[SeqRecord], source_file: str) -> list[Sequence]:
        """
        Convert Biopython SeqRecord objects to NYX Sequence objects.
        Skips records with empty sequences and logs a warning.

        Args:
            records:     List of SeqRecord from Biopython.
            source_file: Original file path string, stored in metadata.

        Returns:
            List of valid Sequence objects.
        """
        sequences: list[Sequence] = []

        for record in records:
            seq_str = str(record.seq).strip()

            if not seq_str:
                logger.warning("Skipping empty sequence: id=%s", record.id)
                continue

            metadata: dict = {"source_file": source_file}

            ec = self._extract_ec(record.description)
            if ec:
                metadata["ec_number"] = ec
                logger.debug("EC number found for %s: %s", record.id, ec)

            sequences.append(
                Sequence(
                    id=record.id,
                    sequence=seq_str,
                    description=record.description,
                    metadata=metadata,
                )
            )

        return sequences

    @staticmethod
    def _extract_ec(header: str) -> Optional[str]:
        """
        Extract the first EC number from a FASTA header string.

        Recognises:
            EC:1.2.3.4          (standard UniProt style)
            ec_number=1.2.3.4   (common annotation style)

        Args:
            header: Full FASTA description line.

        Returns:
            EC number string like "1.2.3.4", or None if not found.
        """
        for pattern in _EC_PATTERNS:
            match = pattern.search(header)
            if match:
                return match.group(1)
        return None


# ---------------------------------------------------------------------------
# Example usage (run this file directly to smoke-test)
# ---------------------------------------------------------------------------
#
#   python core/sequence/loader.py
#
# Expected output with the inline test FASTA:
#   Loaded 3 sequences.
#   Sequence(id='seq1', len=30, ec='3.1.1.4')
#   Sequence(id='seq2', len=30, ec='2.7.1.1')
#   Sequence(id='seq3', len=30, ec=none)
#
if __name__ == "__main__":
    import tempfile, os

    logging.basicConfig(level=logging.DEBUG)

    SAMPLE_FASTA = """\
>seq1 hypothetical protein EC:3.1.1.4 [Soil metagenome]
ATGCGATCGATCGATCGATCGATCGATCGA
>seq2 putative kinase ec_number=2.7.1.1 [Estuary sample]
GCTAGCTAGCTAGCTAGCTAGCTAGCTAGC
>seq3 unknown protein [Sample XYZ]
TTTTAAAACCCCGGGGTTTTAAAACCCCGG
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".fasta", delete=False
    ) as tmp:
        tmp.write(SAMPLE_FASTA)
        tmp_path = tmp.name

    try:
        loader = SequenceLoader()
        seqs = loader.load(tmp_path)
        print(f"\nLoaded {len(seqs)} sequences.")
        for s in seqs:
            print(s)
    finally:
        os.unlink(tmp_path)
