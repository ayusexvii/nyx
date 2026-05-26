"""Advanced feature extraction with k-mers AND physicochemical properties."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.sequence.loader import SequenceLoader
from core.features.kmers import KmerExtractor

# Amino acid properties for physicochemical features
AMINO_ACID_MASSES = {
    'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.16,
    'E': 147.13, 'Q': 146.15, 'G': 75.07, 'H': 155.16, 'I': 131.18,
    'L': 131.18, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
    'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15
}

# Kyte-Doolittle hydrophobicity (positive = hydrophobic)
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

def extract_physicochemical(sequence):
    """Extract physicochemical features from protein sequence."""
    
    seq = sequence.upper()
    length = len(seq)
    
    if length == 0:
        return np.zeros(15)
    
    # 1. Molecular weight
    weight = sum(AMINO_ACID_MASSES.get(aa, 0) for aa in seq)
    
    # 2. Net charge (R + K minus D + E)
    positive = sum(1 for aa in seq if aa in 'RK')
    negative = sum(1 for aa in seq if aa in 'DE')
    net_charge = positive - negative
    
    # 3. Hydrophobicity
    hydro_values = [HYDROPHOBICITY.get(aa, 0) for aa in seq]
    mean_hydro = np.mean(hydro_values)
    std_hydro = np.std(hydro_values)
    pct_hydrophobic = sum(1 for v in hydro_values if v > 0) / length
    
    # 4. Aromaticity (Y, F, W)
    aromatic = sum(1 for aa in seq if aa in 'YFW') / length
    
    # 5. Flexibility (simplified scale)
    flexible_aas = set('GASCTP')
    flexibility = sum(1 for aa in seq if aa in flexible_aas) / length
    
    # 6. Secondary structure propensity (simplified)
    helix_aas = set('EALMKQRH')
    sheet_aas = set('V I Y C W F T'.split())
    coil_aas = set('GNPSD')
    
    helix_prop = sum(1 for aa in seq if aa in helix_aas) / length
    sheet_prop = sum(1 for aa in seq if aa in sheet_aas) / length
    coil_prop = sum(1 for aa in seq if aa in coil_aas) / length
    
    # 7. Composition
    composition = [seq.count(aa) / length for aa in 'ACDEFGHIKLMNPQRSTVWY']
    
    # Combine all features
    features = [
        weight / 1000,  # Normalized molecular weight
        net_charge,
        mean_hydro,
        std_hydro,
        pct_hydrophobic,
        aromatic,
        flexibility,
        helix_prop,
        sheet_prop,
        coil_prop
    ] + composition
    
    return np.array(features)

def build_advanced_features(fasta_file, max_sequences=None):
    """Build feature matrix with k-mers AND physicochemical features."""
    
    print(f"\n📂 Loading {fasta_file}...")
    loader = SequenceLoader()
    all_seqs = loader.load(fasta_file)
    
    # Filter for sequences with EC numbers
    seqs = [s for s in all_seqs if s.metadata.get('ec_number')]
    print(f"✅ Found {len(seqs)} sequences with EC numbers")
    
    if max_sequences and len(seqs) > max_sequences:
        seqs = seqs[:max_sequences]
        print(f"📊 Using first {max_sequences} sequences")
    
    # Extract features
    print("\n🔬 Extracting k-mer features...")
    kmer_extractor = KmerExtractor(k_values=[1, 2], seq_type='protein')
    
    print("🧬 Extracting physicochemical features...")
    
    features_list = []
    labels = []
    
    for i, seq in enumerate(seqs):
        # K-mer features
        kmer_features = kmer_extractor.extract(seq)
        
        # Physicochemical features
        physico_features = extract_physicochemical(seq.sequence)
        
        # Combine
        combined = np.concatenate([kmer_features, physico_features])
        features_list.append(combined)
        
        # Label (EC class first digit)
        ec_full = seq.metadata['ec_number']
        ec_class = ec_full.split('.')[0]
        labels.append(ec_class)
        
        if (i + 1) % 500 == 0:
            print(f"   Processed {i+1}/{len(seqs)} sequences")
    
    X = np.array(features_list)
    y = np.array(labels)
    
    print(f"\n✅ Feature matrix: {X.shape}")
    print(f"   - K-mer features: {len(kmer_features)}")
    print(f"   - Physicochemical: {len(physico_features)}")
    print(f"   - Total: {X.shape[1]}")
    print(f"\n📊 Class distribution:")
    for cls in sorted(np.unique(y)):
        print(f"   EC {cls}: {np.sum(y == cls)} sequences")
    
    # Save
    Path("data/processed").mkdir(exist_ok=True)
    np.save('data/processed/X_advanced.npy', X)
    np.save('data/processed/y_advanced.npy', y)
    
    return X, y

if __name__ == "__main__":
    print("=" * 60)
    print("Building Advanced Feature Matrix (K-mers + Physicochemical)")
    print("=" * 60)
    
    X, y = build_advanced_features('data/external/training_10k.fasta', max_sequences=10000)
    
    print(f"\n✅ Saved to data/processed/X_advanced.npy")
    print(f"✅ Saved to data/processed/y_advanced.npy")
