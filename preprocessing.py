import pandas as pd

def read_fasta(filepath):
    try:
        sequences = {}
        with open(filepath, 'r') as f:
            name = None
            seq = ''
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if name:
                        sequences[name] = seq
                    name = line[1:]
                    seq = ''
                else:
                    seq += line
            if name:
                sequences[name] = seq
        return sequences
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def clean_sequence(sequence):
    return ''.join(c.upper() for c in sequence if c in 'AUCG')

def extract_rna_type(name):
    parts = name.split()
    return parts[0] , parts[1]

def process_fasta_data(fasta_data):
    processed_data = []
    for name, seq in fasta_data.items():
        cleaned_seq = clean_sequence(seq)
        name , rna_type = extract_rna_type(name)
        processed_data.append((cleaned_seq, name,rna_type))
    return processed_data

def load_and_process_fasta(filepath):
    raw_data = read_fasta(filepath)
    if raw_data is None:
        return None
    processed_data = process_fasta_data(raw_data)
    return pd.DataFrame(processed_data, columns=['Sequence','Name','RNA_Type'])