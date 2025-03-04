import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import networkx as nx
from sklearn.model_selection import train_test_split

def read_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        sequence = ""
        current_class = None
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if sequence:
                    sequences.append({'sequence': sequence, 'class': current_class})
                current_class = line.split()[1]
                sequence = ""
            else:
                sequence += line
                
        if sequence:
            sequences.append({'sequence': sequence, 'class': current_class})
    return sequences

def one_hot_encode(sequence):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, 'N': 4, '-': 4}
    seq_len = len(sequence)
    one_hot = np.zeros((seq_len, 5), dtype=np.float32)
    
    for i, nucleotide in enumerate(sequence):
        if nucleotide in mapping:
            one_hot[i, mapping[nucleotide]] = 1.0
        else:
            one_hot[i, 4] = 1.0  # Unknown nucleotide
            
    return one_hot

def sequence_to_graph(sequence, k=4, stride=1):
    one_hot = one_hot_encode(sequence)
    seq_len = len(sequence)
    
    # Create node features (k-mer embeddings)
    node_features = []
    node_indices = []
    
    for i in range(0, seq_len - k + 1, stride):
        kmer = sequence[i:i+k]
        if len(kmer) == k:
            node_features.append(one_hot[i:i+k].flatten())
            node_indices.append(i)
    
    if not node_features:
        # Handle sequences shorter than k
        node_features = [one_hot.flatten()]
        node_indices = [0]
    
    # Create edges based on sequence proximity
    edges = []
    for i in range(len(node_indices)):
        for j in range(i+1, len(node_indices)):
            if abs(node_indices[i] - node_indices[j]) <= 5:  # Connect nodes within 5 positions
                edges.append([i, j])
                edges.append([j, i])  # Add reverse edge for undirected graph
    
    # If no edges, create self-loops
    if not edges and len(node_indices) > 0:
        edges = [[i, i] for i in range(len(node_indices))]
    
    node_features = torch.tensor(np.array(node_features), dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
    
    return Data(x=node_features, edge_index=edge_index)

class RNAGraphDataset(Dataset):
    def __init__(self, sequences, class_to_idx=None):
        self.sequences = sequences
        
        if class_to_idx is None:
            classes = sorted(list(set(seq['class'] for seq in sequences)))
            self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx
            
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_data = self.sequences[idx]
        sequence = seq_data['sequence']
        label = self.class_to_idx[seq_data['class']]
        
        graph = sequence_to_graph(sequence)
        graph.y = torch.tensor(label, dtype=torch.long)
        
        return graph
    
    @staticmethod
    def collate_fn(batch):
        return Batch.from_data_list(batch)

def load_data(train_file_path, test_file_path):
    train_sequences = read_fasta(train_file_path)
    test_sequences = read_fasta(test_file_path)
    
    # Create datasets
    train_dataset = RNAGraphDataset(train_sequences)
    test_dataset = RNAGraphDataset(test_sequences)
    
    return train_dataset, test_dataset

def create_dataloader(train_dataset, test_dataset, batch_size=32, num_workers=4):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=RNAGraphDataset.collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=RNAGraphDataset.collate_fn
    )
    
    return train_loader, test_loader  # Return both loaders 