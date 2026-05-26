"""Advanced training with ensemble methods for 90%+ accuracy."""

import sys
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

def train_advanced():
    """Train ensemble model for high accuracy."""
    
    print("=" * 60)
    print("Project NYX - Advanced Training (Target: 90%+ Accuracy)")
    print("=" * 60)
    
    # Load advanced features
    print("\n1. Loading advanced features...")
    X = np.load('data/processed/X_advanced.npy')
    y = np.load('data/processed/y_advanced.npy')
    
    print(f"   Feature matrix: {X.shape}")
    print(f"   Number of classes: {len(np.unique(y))}")
    
    # Scale features for SVM
    print("\n2. Preprocessing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    print("\n3. Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training: {len(X_train)} sequences")
    print(f"   Testing: {len(X_test)} sequences")
    
    # Train multiple models
    print("\n4. Training models...")
    
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ),
        'SVM': SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            random_state=42
        )
    }
    
    results = {}
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        print(f"\n   Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
        
        print(f"      ✅ {name} accuracy: {accuracy:.3f}")
    
    # Cross-validation on best model
    print("\n5. Cross-validation on best model...")
    cv_scores = cross_val_score(best_model, X_scaled, y, cv=5)
    print(f"   Cross-validation scores: {cv_scores}")
    print(f"   ✅ Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    # Detailed report
    print("\n6. Detailed classification report:")
    y_pred_best = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_best))
    
    # Confusion matrix
    print("\n7. Confusion matrix:")
    cm = confusion_matrix(y_test, y_pred_best)
    print(cm)
    
    # Feature importance (RandomForest only)
    if hasattr(best_model, 'feature_importances_'):
        print("\n8. Top 20 most important features:")
        importance = best_model.feature_importances_
        top_idx = np.argsort(importance)[-20:][::-1]
        for i, idx in enumerate(top_idx):
            print(f"   {i+1}. Feature {idx}: {importance[idx]:.4f}")
    
    # Save model and scaler
    import joblib
    Path("models").mkdir(exist_ok=True)
    joblib.dump(best_model, 'models/ec_classifier_advanced.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("\n" + "=" * 60)
    print(f"✅ BEST MODEL: {max(results, key=results.get)}")
    print(f"✅ Test Accuracy: {best_score:.3f}")
    print(f"✅ Cross-validation: {cv_scores.mean():.3f}")
    print("=" * 60)
    print("\n✅ Model saved to models/ec_classifier_advanced.pkl")
    print("✅ Scaler saved to models/scaler.pkl")
    
    return best_model, best_score

if __name__ == "__main__":
    train_advanced()
