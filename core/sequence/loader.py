"""
Sequence loader for Project NYX.
Handles FASTA (plain and gzipped) with EC number extraction from headers.
"""

import gzip
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from Bio import SeqIO

logger = logging.getLogger(__name__)

# Matches EC number patterns in headers
_EC_PATTERNS = [
    re.compile(r"EC:(\d+\.\d+\.\d+\.\d+)", re.IGNORECASE),
    re.compile(r"ec_number=(\d+\.\d+\.\d+\.\d+)", re.IGNORECASE),
    re.compile(r"\[EC:(\d+\.\d+\.\d+\.\d+)\]", re.IGNORECASE),
]


@dataclass
class Sequence:
    """Represents a single biological sequence with parsed metadata."""
    id: str
    sequence: str
    description: str = ""
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.sequence)

    def __repr__(self) -> str:
        ec = self.metadata.get("ec_number", "none")
        return f"Sequence(id={self.id!r}, len={len(self)}, ec={ec!r})"


class SequenceLoader:
    """Loads biological sequences from FASTA files (plain or gzipped)."""

    SUPPORTED_FORMATS = {
        ".fasta": "fasta",
        ".fa": "fasta",
        ".fna": "fasta",
        ".faa": "fasta",
        ".fastq": "fastq",
        ".fq": "fastq",
    }

    def load(self, filepath: str | Path) -> List[Sequence]:
        """Load sequences from a FASTA file."""
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Sequence file not found: {path}")

        fmt = self._detect_format(path)
        logger.info("Loading sequences from %s (format=%s)", path.name, fmt)

        records = self._parse(path, fmt)
        sequences = self._convert(records, source_file=str(path))

        logger.info("Loaded %d sequences from %s", len(sequences), path.name)
        return sequences

    def _detect_format(self, path: Path) -> str:
        """Resolve Bio format string from file extension."""
        name = path.name
        if name.endswith(".gz"):
            name = name[:-3]
        ext = Path(name).suffix.lower()

        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file extension '{ext}'. "
                f"Supported: {list(self.SUPPORTED_FORMATS)}"
            )

        # For FASTA files, use fasta-blast which handles comment lines
        if self.SUPPORTED_FORMATS[ext] == "fasta":
            return "fasta-blast"

        return self.SUPPORTED_FORMATS[ext]

    def _parse(self, path: Path, fmt: str) -> List:
        """Parse the file with Biopython."""
        if path.name.endswith(".gz"):
            handle = gzip.open(path, "rt")
        else:
            handle = open(path, "r")

        records = list(SeqIO.parse(handle, fmt))
        handle.close()
        return records

    def _convert(self, records, source_file: str) -> List[Sequence]:
        """Convert Biopython records to Sequence objects."""
        sequences = []

        for record in records:
            seq_str = str(record.seq).strip()
            if not seq_str:
                continue

            metadata = {"source_file": source_file}

            # Extract EC number from header
            for pattern in _EC_PATTERNS:
                match = pattern.search(record.description)
                if match:
                    metadata["ec_number"] = match.group(1)
                    break

            sequences.append(Sequence(
                id=record.id,
                sequence=seq_str,
                description=record.description,
                metadata=metadata
            ))

        return sequences


# Example usage
if __name__ == "__main__":
    import tempfile, os

    logging.basicConfig(level=logging.INFO)

    SAMPLE_FASTA = """\
>seq1 hypothetical protein EC:3.1.1.4 [Soil metagenome]
ATGCGATCGATCGATCGATCGATCGATCGA
>seq2 putative kinase ec_number=2.7.1.1 [Estuary sample]
GCTAGCTAGCTAGCTAGCTAGCTAGCTAGC
>seq3 unknown protein [Sample XYZ]
TTTTAAAACCCCGGGGTTTTAAAACCCCGG
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as tmp:
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
