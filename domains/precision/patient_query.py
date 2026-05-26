"""Patient query tool for drug response prediction."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from domains.precision.variant_loader import VariantLoader, GeneticVariant
from domains.precision.drug_predictor import DrugResponsePredictor

def main():
    print("=" * 50)
    print("Project NYX - Precision Medicine Advisor")
    print("=" * 50)
    
    # Load and train
    loader = VariantLoader()
    variants = loader.load_demo_data('domains/precision/data/demo_variants.csv')
    
    predictor = DrugResponsePredictor()
    predictor.train(variants)
    
    print("\nAvailable drugs: clopidogrel, codeine, warfarin, fluorouracil")
    print("Available genes: CYP2C19, CYP2D6, VKORC1, CYP2C9, DPYD\n")
    
    while True:
        print("-" * 40)
        gene = input("Enter gene (or 'quit'): ").strip().upper()
        if gene.lower() == 'quit':
            break
        
        variant = input("Enter variant (e.g., *2, *4, *17): ").strip()
        drug = input("Enter drug: ").strip().lower()
        
        # Create patient variant
        patient_variant = GeneticVariant(
            gene=gene,
            variant=variant,
            drug=drug,
            response="unknown"
        )
        
        # Predict
        prediction = predictor.predict(patient_variant)
        
        print(f"\n📊 Result for {gene}{variant} taking {drug}:")
        print(f"   Expected response: {prediction.upper()}")
        
        # Add clinical recommendation
        if prediction == 'poor/adverse':
            print(f"   ⚠️  Consider alternative medication or dose adjustment")
        elif prediction == 'rapid':
            print(f"   📈 May require higher dose")
        else:
            print(f"   ✅ Standard dosing recommended")
        print()

if __name__ == "__main__":
    main()
