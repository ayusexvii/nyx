"""Download REAL enzyme data with EC numbers from UniProt."""

import requests
import time
from pathlib import Path

def download_ec_batch(ec_class, max_sequences=2000, retry=3):
    """Download sequences for a specific EC class with retry logic."""
    
    # Use the working endpoint
    url = "https://rest.uniprot.org/uniprotkb/stream"
    
    params = {
        "format": "fasta",
        "query": f"ec:{ec_class}.* AND reviewed:true",
        "size": max_sequences
    }
    
    print(f"\n📥 Downloading EC {ec_class}.* (max {max_sequences} sequences)...")
    
    for attempt in range(retry):
        try:
            response = requests.get(url, params=params, timeout=60)
            
            if response.status_code == 200:
                content = response.text
                
                if content and content.startswith('>'):
                    # Count sequences
                    count = content.count('>')
                    
                    # Save to file
                    output_file = f"data/external/ec_{ec_class}_raw.fasta"
                    with open(output_file, 'w') as f:
                        f.write(content)
                    
                    print(f"   ✅ Downloaded {count} sequences to {output_file}")
                    return output_file
                else:
                    print(f"   ⚠️ No sequences found for EC {ec_class}")
                    return None
            else:
                print(f"   ⚠️ Attempt {attempt+1} failed: HTTP {response.status_code}")
                time.sleep(2)
                
        except Exception as e:
            print(f"   ⚠️ Attempt {attempt+1} error: {e}")
            time.sleep(2)
    
    print(f"   ❌ Failed to download EC {ec_class} after {retry} attempts")
    return None

def fix_ec_headers(input_file, output_file):
    """Add EC numbers to FASTA headers by extracting from description."""
    
    print(f"\n🔧 Fixing headers in {input_file}...")
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if line.startswith('>'):
                # Try to find EC number in the description
                import re
                ec_match = re.search(r'EC[:_\s]+(\d+\.\d+\.\d+\.\d+)', line, re.IGNORECASE)
                if ec_match:
                    ec = ec_match.group(1)
                    # Add EC to header in our standard format
                    new_header = line.rstrip() + f" [EC:{ec}]\n"
                    outfile.write(new_header)
                else:
                    outfile.write(line)
            else:
                outfile.write(line)
    
    # Count fixed sequences
    with open(output_file, 'r') as f:
        count = sum(1 for line in f if line.startswith('>') and '[EC:' in line)
    
    print(f"   ✅ Fixed {count} headers with EC numbers")
    return output_file

if __name__ == "__main__":
    print("=" * 60)
    print("Downloading REAL Enzyme Data for 90%+ Accuracy")
    print("=" * 60)
    
    Path("data/external").mkdir(parents=True, exist_ok=True)
    
    # Download multiple EC classes
    ec_classes = [1, 2, 3, 4, 5, 6]  # EC classes 1-6
    
    downloaded_files = []
    for ec in ec_classes:
        raw_file = download_ec_batch(ec, max_sequences=2000)
        if raw_file:
            fixed_file = raw_file.replace('_raw', '_fixed')
            fix_ec_headers(raw_file, fixed_file)
            downloaded_files.append(fixed_file)
        time.sleep(1)  # Be nice to the API
    
    # Combine all files
    print("\n" + "=" * 60)
    print("Combining all downloaded data...")
    print("=" * 60)
    
    combined_file = "data/external/training_10k.fasta"
    with open(combined_file, 'w') as outfile:
        for file in downloaded_files:
            with open(file, 'r') as infile:
                outfile.write(infile.read())
    
    # Count total sequences
    with open(combined_file, 'r') as f:
        total = sum(1 for line in f if line.startswith('>'))
    
    print(f"\n✅ Combined dataset: {total} sequences saved to {combined_file}")
    
    # Show sample
    print("\n📋 Sample headers:")
    with open(combined_file, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('>') and i < 5:
                print(f"   {line.strip()[:80]}...")
    
    print("\n" + "=" * 60)
    print("Download complete! Now run: python build_features_advanced.py")
    print("=" * 60)
