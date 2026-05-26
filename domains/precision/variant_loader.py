"""Variant loader for precision medicine module."""

import csv
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class GeneticVariant:
    """Represents a human genetic variant."""
    gene: str
    variant: str
    drug: str
    response: str
    confidence: str = "high"
    evidence: str = "clinical"

class VariantLoader:
    """Load pharmacogenomic variant data."""
    
    def load_demo_data(self, filepath: str) -> List[GeneticVariant]:
        """Load demo variant-drug response data."""
        variants = []
        
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                variant = GeneticVariant(
                    gene=row['gene'],
                    variant=row['variant'],
                    drug=row['drug'],
                    response=row['response']
                )
                variants.append(variant)
        
        return variants
    
    def get_variant_features(self, variant: GeneticVariant) -> Dict:
        """Convert variant to feature dictionary."""
        features = {
            'gene': variant.gene,
            'variant_type': self._classify_variant(variant.variant),
            'drug_class': self._get_drug_class(variant.drug),
            'response_score': self._response_to_score(variant.response)
        }
        return features
    
    def _classify_variant(self, variant: str) -> str:
        """Classify variant type."""
        if '*' in variant:
            return 'star_allele'
        elif '>' in variant:
            return 'snv'
        elif 'del' in variant or 'ins' in variant:
            return 'indel'
        return 'unknown'
    
    def _get_drug_class(self, drug: str) -> str:
        """Categorize drug type."""
        drug_classes = {
            'clopidogrel': 'antiplatelet',
            'codeine': 'opioid',
            'warfarin': 'anticoagulant',
            'fluorouracil': 'chemotherapy'
        }
        return drug_classes.get(drug, 'other')
    
    def _response_to_score(self, response: str) -> int:
        """Convert response to numeric score."""
        scores = {
            'ultrarapid': 4,
            'rapid': 3,
            'normal': 2,
            'intermediate': 1,
            'poor': 0,
            'sensitive': 0,
            'toxicity': -1
        }
        return scores.get(response.lower(), 2)

# Quick test
if __name__ == "__main__":
    loader = VariantLoader()
    variants = loader.load_demo_data('domains/precision/data/demo_variants.csv')
    print(f"Loaded {len(variants)} variants")
    for v in variants[:3]:
        features = loader.get_variant_features(v)
        print(f"  {v.gene} {v.variant} -> {v.drug}: {v.response}")
