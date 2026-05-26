"""Drug response predictor for precision medicine."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

class DrugResponsePredictor:
    """Predict drug response from genetic variants."""
    
    def __init__(self):
        self.model = None
        self.variant_encoder = {}
    
    def prepare_features(self, variants, response_labels=None):
        """Convert variant data to feature matrix."""
        
        # Simple encoding for demo
        # In production: one-hot encode genes, drugs, variant types
        
        gene_map = {'CYP2C19': 0, 'CYP2D6': 1, 'VKORC1': 2, 'CYP2C9': 3, 'DPYD': 4}
        drug_map = {'clopidogrel': 0, 'codeine': 1, 'warfarin': 2, 'fluorouracil': 3}
        
        X = []
        y = []
        
        for v in variants:
            features = [
                gene_map.get(v.gene, 0),
                drug_map.get(v.drug, 0),
                1 if '*' in v.variant else 0,  # star allele
                1 if '>' in v.variant else 0,  # SNV
            ]
            X.append(features)
            
            if response_labels:
                y.append(self._response_score(v.response))
        
        X = np.array(X)
        
        if y:
            y = np.array(y)
            return X, y
        return X
    
    def _response_score(self, response: str) -> int:
        """Convert response to binary or multiclass."""
        if response in ['poor', 'sensitive', 'toxicity']:
            return 0  # poor response / adverse
        elif response in ['rapid', 'ultrarapid']:
            return 2  # rapid response
        else:
            return 1  # normal response
    
    def train(self, variants):
        """Train model on variant-response data."""
        X, y = self.prepare_features(variants, response_labels=True)
        
        print(f"Training on {len(X)} samples with {X.shape[1]} features")
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        accuracy = self.model.score(X, y)
        print(f"Training accuracy: {accuracy:.3f}")
        
        return self.model
    
    def predict(self, variant):
        """Predict response for a single variant."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X = self.prepare_features([variant])
        pred = self.model.predict(X)[0]
        
        response_map = {0: 'poor/adverse', 1: 'normal', 2: 'rapid'}
        return response_map.get(pred, 'unknown')
    
    def save(self, path: str):
        """Save model to file."""
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from file."""
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")

# Demo
if __name__ == "__main__":
    from variant_loader import VariantLoader
    
    # Load data
    loader = VariantLoader()
    variants = loader.load_demo_data('data/demo_variants.csv')
    
    # Train predictor
    predictor = DrugResponsePredictor()
    predictor.train(variants)
    
    # Test prediction
    test_variant = variants[0]
    pred = predictor.predict(test_variant)
    print(f"\nTest prediction for {test_variant.gene} {test_variant.variant}")
    print(f"  Drug: {test_variant.drug}")
    print(f"  Expected: {test_variant.response}")
    print(f"  Predicted: {pred}")
