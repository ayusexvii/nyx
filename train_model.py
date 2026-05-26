"""Train enzyme EC class predictor for Project NYX."""

import sys
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).parent))

def train_model(test_size=0.2, use_cross_validation=True):
    """Train RandomForest classifier on k-mer features."""
    
    print("=" * 50)
    print("Project NYX - Training EC Class Predictor")
    print("=" * 50)
    
    # Check if features exist
    if not Path("data/processed/X.npy").exists():
        print("\n❌ Features not found. Run build_features.py first!")
        return None, 0
    
    # Load features
    print("\n1. Loading features...")
    X = np.load('data/processed/X.npy')
    y = np.load('data/processed/y.npy')
    
    print(f"   Feature matrix: {X.shape}")
    print(f"   Classes: {np.unique(y)}")
    print(f"   Class distribution:")
    for cls in np.unique(y):
        print(f"      EC {cls}: {np.sum(y == cls)} sequences")
    
    # For small datasets, use cross-validation instead of train/test split
    if len(X) < 20:
        print("\n⚠️ Small dataset detected. Using cross-validation instead of train/test split.")
        use_cross_validation = True
    
    if use_cross_validation:
        print("\n2. Using cross-validation...")
        from sklearn.model_selection import cross_val_score
        
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Perform 3-fold cross-validation (minimum for small dataset)
        cv_folds = min(3, len(np.unique(y)))
        scores = cross_val_score(model, X, y, cv=cv_folds)
        
        print(f"   Cross-validation scores: {scores}")
        print(f"   ✅ Mean accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
        
        # Train final model on all data
        print("\n3. Training final model on all data...")
        model.fit(X, y)
        
        print("\n4. Feature importance (top 10):")
        importance = model.feature_importances_
        top_idx = np.argsort(importance)[-10:][::-1]
        for i, idx in enumerate(top_idx):
            print(f"   {i+1}. Feature {idx}: {importance[idx]:.4f}")
        
        # Save model
        Path("models").mkdir(exist_ok=True)
        import joblib
        joblib.dump(model, 'models/ec_classifier.pkl')
        print("\n✅ Model saved to models/ec_classifier.pkl")
        
        return model, scores.mean()
    
    else:
        # Original train/test split for larger datasets
        print("\n2. Splitting train/test...")
        
        # Adjust test size to ensure at least one sample per class
        n_classes = len(np.unique(y))
        min_test_size = max(n_classes, 2) / len(X)
        actual_test_size = max(test_size, min_test_size)
        
        print(f"   Adjusted test size: {actual_test_size:.2f} (was {test_size})")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=actual_test_size, random_state=42, stratify=y
        )
        print(f"   Training: {len(X_train)} sequences")
        print(f"   Testing: {len(X_test)} sequences")
        
        # Train model
        print("\n3. Training RandomForest...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        print("\n4. Evaluating...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n   ✅ Test Accuracy: {accuracy:.3f}")
        
        print("\n5. Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        Path("models").mkdir(exist_ok=True)
        import joblib
        joblib.dump(model, 'models/ec_classifier.pkl')
        print("\n✅ Model saved to models/ec_classifier.pkl")
        
        return model, accuracy

if __name__ == "__main__":
    train_model()
