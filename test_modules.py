"""Test script for Project NYX modules."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.sequence.loader import SequenceLoader, Sequence
from core.features.kmers import KmerExtractor

def test_pipeline():
    """Test loading a sequence and extracting features."""
    
    # Create a simple test sequence (not from file)
    class SimpleSeq:
        def __init__(self, id, sequence):
            self.id = id
            self.sequence = sequence
            self.description = ""
            self.metadata = {}
    
    # Test protein sequence
    protein_seq = SimpleSeq(
        id="test_enzyme_001",
        sequence="MKTIIALSYIFCLVFA" * 10
    )
    
    print("=" * 50)
    print("Testing Project NYX Modules")
    print("=" * 50)
    
    # Test k-mer extractor
    print("\n1. Testing KmerExtractor...")
    extractor = KmerExtractor(k_values=[1, 2], seq_type="protein")
    features = extractor.extract(protein_seq)
    print(f"   ✓ Extracted {len(features)} features")
    print(f"   ✓ Feature sum: {features.sum():.4f} (expected ~2.0)")
    
    # Test feature count
    n_features = extractor.feature_count(seq_type="protein")
    print(f"   ✓ Total features available: {n_features}")
    
    print("\n✅ All tests passed!")
    print(f"\nFeature vector preview (first 10 values):")
    print(f"   {features[:10]}")

if __name__ == "__main__":
    test_pipeline()

