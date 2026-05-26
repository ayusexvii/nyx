import requests
from pathlib import Path

def download_ec_class(ec_number, max_sequences=500):
    """Download sequences for a specific EC number."""
    
    query = f"ec:{ec_number}"
    url = "https://rest.uniprot.org/uniprotkb/stream"
    
    params = {
        "format": "fasta",
        "query": query,
        "size": max_sequences
    }
    
    print(f"\nDownloading EC {ec_number}...")
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None
    
    content = response.text
    
    if not content or not content.startswith('>'):
        print(f"No sequences found for EC {ec_number}")
        return None
    
    output_file = f"data/external/ec_{ec_number.replace('.', '_')}.fasta"
    Path("data/external").mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write(content)
    
    # Count sequences
    count = content.count('>')
    print(f"✅ Saved {count} sequences to {output_file}")
    
    return output_file

def combine_files():
    """Combine all EC files into one training file."""
    files = list(Path("data/external").glob("ec_*.fasta"))
    
    if not files:
        print("No EC files found to combine")
        return
    
    output_file = "data/external/training_data.fasta"
    
    with open(output_file, 'w') as outfile:
        for f in files:
            with open(f, 'r') as infile:
                outfile.write(infile.read())
    
    # Count total sequences
    with open(output_file, 'r') as f:
        count = sum(1 for line in f if line.startswith('>'))
    
    print(f"\n✅ Combined {len(files)} files into {output_file}")
    print(f"   Total sequences: {count}")
    
    return output_file

if __name__ == "__main__":
    print("="*50)
    print("Downloading EC class sequences")
    print("="*50)
    
    # Download specific EC numbers
    download_ec_class("1.1.1.1", max_sequences=500)   # Alcohol dehydrogenase
    download_ec_class("1.1.1.27", max_sequences=500)  # Lactate dehydrogenase
    download_ec_class("2.7.1.1", max_sequences=500)   # Kinase
    download_ec_class("2.7.1.2", max_sequences=500)   # Glucokinase
    download_ec_class("3.4.21.4", max_sequences=500)  # Trypsin
    download_ec_class("3.4.21.5", max_sequences=500)  # Thrombin
    download_ec_class("4.2.1.1", max_sequences=500)   # Carbonic anhydrase
    download_ec_class("5.1.1.1", max_sequences=500)   # Alanine racemase
    
    print("\n" + "="*50)
    print("Download complete!")
    print("="*50)
    
    # Combine all files
    combine_files()
