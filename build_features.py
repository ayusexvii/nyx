"""Build feature matrix for training Project NYX."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.sequence.loader import SequenceLoader
from core.features.kmers import KmerExtractor

def build_feature_matrix(fasta_file, max_sequences=None):
    """Extract k-mer features from sequences with EC numbers."""
    
    print(f"Loading {fasta_file}...")
    loader = SequenceLoader()
    all_seqs = loader.load(fasta_file)
    
    # Filter for sequences with EC numbers
    seqs = [s for s in all_seqs if s.metadata.get('ec_number')]
    print(f"Sequences with EC numbers: {len(seqs)}")
    
    if max_sequences and len(seqs) > max_sequences:
        seqs = seqs[:max_sequences]
        print(f"Using first {max_sequences} sequences")
    
    # Extract features
    print("Extracting k-mer features...")
    extractor = KmerExtractor(k_values=[1, 2], seq_type='protein')
    
    features = []
    labels = []
    ids = []
    
    for i, seq in enumerate(seqs):
        feat = extractor.extract(seq)
        features.append(feat)
        
        # Use EC class (first digit only: 1,2,3,4,5,6,7)
        ec_full = seq.metadata['ec_number']
        ec_class = ec_full.split('.')[0]
        labels.append(ec_class)
        ids.append(seq.id)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(seqs)} sequences")
    
    X = np.array(features)
    y = np.array(labels)
    
    print(f"\n✅ Feature matrix: {X.shape}")
    print(f"✅ Unique EC classes: {np.unique(y)}")
    
    # Save to processed directory
    Path("data/processed").mkdir(exist_ok=True)
    np.save('data/processed/X.npy', X)
    np.save('data/processed/y.npy', y)
    
    return X, y, ids

if __name__ == "__main__":
    X, y, ids = build_feature_matrix('data/external/enzymes_small.fasta')
    print(f"\n✅ Saved to data/processed/")
