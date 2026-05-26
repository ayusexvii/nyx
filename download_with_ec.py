"""Download sequences with EC numbers in headers using JSON API."""

import requests
from pathlib import Path

def download_ec_with_headers(ec_number, max_sequences=500):
    """Download sequences and create FASTA with EC in header."""
    
    url = "https://rest.uniprot.org/uniprotkb/search"
    
    params = {
        "query": f"ec:{ec_number}",
        "format": "json",
        "size": max_sequences,
        "fields": "accession,sequence,ec_number,protein_name"
    }
    
    print(f"\nDownloading EC {ec_number}...")
    
    response = requests.get(url, params=params)
    data = response.json()
    
    results = data.get('results', [])
    print(f"Found {len(results)} sequences")
    
    if not results:
        print(f"No sequences found for EC {ec_number}")
        return None
    
    output_file = f"data/external/ec_{ec_number.replace('.', '_')}.fasta"
    Path("data/external").mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(output_file, 'w') as f:
        for entry in results:
            accession = entry.get('primaryAccession', '')
            ec_list = entry.get('protein', {}).get('ecNumber', [])
            sequence = entry.get('sequence', {}).get('value', '')
            protein_name = entry.get('protein', {}).get('recommendedName', {}).get('fullName', {}).get('value', '')
            
            if not sequence or not ec_list:
                continue
            
            ec = ec_list[0]
            count += 1
            
            # Write header with EC number
            f.write(f">{accession} {protein_name} [EC:{ec}]\n")
            
            # Write sequence in 60-char lines
            for i in range(0, len(sequence), 60):
                f.write(sequence[i:i+60] + '\n')
    
    print(f"✅ Saved {count} sequences to {output_file}")
    return output_file

if __name__ == "__main__":
    print("="*50)
    print("Downloading EC classes with EC in headers")
    print("="*50)
    
    # Download different EC classes
    download_ec_with_headers("1.1.1.1", max_sequences=500)
    download_ec_with_headers("1.1.1.27", max_sequences=500)
    download_ec_with_headers("2.7.1.1", max_sequences=500)
    download_ec_with_headers("3.4.21.4", max_sequences=500)
    download_ec_with_headers("4.2.1.1", max_sequences=500)
    
    print("\n" + "="*50)
    print("Download complete!")
    print("="*50)
    
    # Combine all files
    from pathlib import Path
    files = list(Path("data/external").glob("ec_*.fasta"))
    
    if files:
        with open("data/external/training_with_ec.fasta", 'w') as outfile:
            for f in files:
                with open(f, 'r') as infile:
                    outfile.write(infile.read())
        
        with open("data/external/training_with_ec.fasta", 'r') as f:
            count = sum(1 for line in f if line.startswith('>'))
        print(f"\n✅ Combined {len(files)} files into training_with_ec.fasta")
        print(f"   Total sequences: {count}")
