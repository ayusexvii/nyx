"""Download real pharmacogenomics data from PharmGKB."""

import requests
import zipfile
import io
from pathlib import Path

def download_pharmgkb_data():
    """Download clinical annotations from PharmGKB."""
    
    # PharmGKB requires registration, so we'll use a public dataset
    # Alternative: Use dbSNP or ClinVar data
    
    print("Note: PharmGKB requires free registration")
    print("Visit: https://www.pharmgkb.org/page/downloads")
    print("\nFor now, using demo data. To get real data:")
    print("1. Register at PharmGKB")
    print("2. Download 'clinical_annotations.tsv'")
    print("3. Place in domains/precision/data/")
    
if __name__ == "__main__":
    download_pharmgkb_data()
