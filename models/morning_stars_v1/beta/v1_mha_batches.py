import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attn(x.permute(1, 0, 2), x.permute(1, 0, 2), x.permute(1, 0, 2))  
        x = self.norm1(x + self.dropout(attn_output.permute(1, 0, 2)))  
        return x


class TCR_Epitope_Dataset(Dataset):
    def __init__(self, data, tcr_batch_files, epitope_batch_files):
        self.data = data
        
        # Store batch file paths
        self.tcr_batch_files = tcr_batch_files
        self.epitope_batch_files = epitope_batch_files

        # Map each TCR to its batch file
        self.tcr_batch_map = self._create_batch_map(self.tcr_batch_files, "tcr")
        self.epitope_batch_map = self._create_batch_map(self.epitope_batch_files, "epitope")

    def _create_batch_map(self, batch_files, prefix):
        """Create a mapping from sample names to batch file paths."""
        mapping = {}
        for batch_file in batch_files:
            batch_data = np.load(batch_file, mmap_mode="r")  # Load metadata without full load
            for key in batch_data.files:
                mapping[key] = batch_file  # Store batch file location
        return mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tcr_id = row['TRB_CDR3']
        epitope_id = row['Epitope']
        
        # Load TCR embedding dynamically
        tcr_batch_file = self.tcr_batch_map[tcr_id]
        tcr_batch_data = np.load(tcr_batch_file, mmap_mode="r")
        tcr_embedding = torch.tensor(tcr_batch_data[tcr_id], dtype=torch.float32)

        # Load Epitope embedding dynamically
        epitope_batch_file = self.epitope_batch_map[epitope_id]
        epitope_batch_data = np.load(epitope_batch_file, mmap_mode="r")
        epitope_embedding = torch.tensor(epitope_batch_data[epitope_id], dtype=torch.float32)

        # Load label
        label = torch.tensor(row['Binding'], dtype=torch.float32)
        
        return tcr_embedding, epitope_embedding, label


class TCR_Epitope_Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, max_tcr_length, max_epitope_length, dropout=0.1):
        super(TCR_Epitope_Transformer, self).__init__()
        self.tcr_embedding = nn.Linear(512, embed_dim)
        # print('embed_dim: ', embed_dim)
        self.epitope_embedding = nn.Linear(512, embed_dim)
        
        self.tcr_positional_encoding = nn.Parameter(torch.randn(1, max_tcr_length, embed_dim))
        # print('max_tcr_length: ',max_tcr_length)
        self.epitope_positional_encoding = nn.Parameter(torch.randn(1, max_epitope_length, embed_dim))
        # print('max_epitope_length: ',max_epitope_length)

        self.transformer_layers = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(embed_dim, 1)

    def forward(self, tcr, epitope):
        tcr = self.tcr_embedding(tcr) + self.tcr_positional_encoding
        epitope = self.epitope_embedding(epitope) + self.epitope_positional_encoding
        
        # combined = torch.cat((tcr.unsqueeze(0), epitope.unsqueeze(0)), dim=0)
        combined = torch.cat((tcr, epitope), dim=1)  # Concatenating along sequence length
  

        for layer in self.transformer_layers:
            combined = layer(combined)

        # pooled = combined[:, 0, :]  # Take the first token (or use other aggregation)
        # output = torch.sigmoid(self.output_layer(pooled)).squeeze(1)  # Ensure shape is (batch_size)

        pooled = combined.mean(dim=1)  # Average across all tokens, shape: (B, D)
        output = torch.sigmoid(self.output_layer(pooled)).squeeze(1)

        return output
