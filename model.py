import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        return x

class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(GIN, self).__init__()
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        nn1 = MLP(input_dim, hidden_dim, hidden_dim)
        self.convs.append(GINConv(nn1, train_eps=True))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for i in range(num_layers - 1):
            nn_i = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.convs.append(GINConv(nn_i, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.mlp = MLP(hidden_dim, hidden_dim, output_dim)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply GIN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            
        # Global pooling
        x = global_add_pool(x, batch)
        
        # Final MLP
        x = self.mlp(x)
        
        return x

class RNAClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3):
        super(RNAClassifier, self).__init__()
        
        self.gin = GIN(input_dim, hidden_dim, hidden_dim, num_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, data):
        x = self.gin(data)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.classifier(x)
        return x 