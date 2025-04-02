import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        # Self-attention for TCR
        self.tcr_self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        # Self-attention for epitope
        self.epitope_self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        # Cross-attention: TCR queries attend to epitope
        self.tcr_cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tcr, epitope, tcr_mask=None, epitope_mask=None):
        # Self-attention for TCR
        tcr_norm = self.norm1(tcr)
        tcr_self, _ = self.tcr_self_attn(
            tcr_norm.permute(1, 0, 2),
            tcr_norm.permute(1, 0, 2),
            tcr_norm.permute(1, 0, 2),
            key_padding_mask=tcr_mask
        )
        tcr = tcr + self.dropout(tcr_self.permute(1, 0, 2))

        # Self-attention for epitope
        epitope_norm = self.norm2(epitope)
        epitope_self, _ = self.epitope_self_attn(
            epitope_norm.permute(1, 0, 2),
            epitope_norm.permute(1, 0, 2),
            epitope_norm.permute(1, 0, 2),
            key_padding_mask=epitope_mask
        )
        epitope = epitope + self.dropout(epitope_self.permute(1, 0, 2))

        # Cross-attention: TCR attends to epitope
        tcr_norm = self.norm3(tcr)
        tcr_cross, _ = self.tcr_cross_attn(
            tcr_norm.permute(1, 0, 2),  # TCR as query
            epitope.permute(1, 0, 2),   # Epitope as key/value
            epitope.permute(1, 0, 2),
            key_padding_mask=epitope_mask
        )
        tcr = tcr + self.dropout(tcr_cross.permute(1, 0, 2))

        # Shared FFN
        tcr = tcr + self.dropout(self.ffn(tcr))
        epitope = epitope + self.dropout(self.ffn(epitope))
        
        return tcr, epitope

# Custom Dataset class to handle lazy-loaded embeddings
class LazyTCR_Epitope_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_frame, tcr_embeddings, epitope_embeddings):
        """
        Args:
            data_frame (DataFrame): The DataFrame containing sample data.
            tcr_embeddings (h5py.File): HDF5 file containing TCR embeddings.
            epitope_embeddings (h5py.File): HDF5 file containing epitope embeddings.
        """
        self.data_frame = data_frame
        self.tcr_embeddings = tcr_embeddings
        self.epitope_embeddings = epitope_embeddings

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sample = self.data_frame.iloc[idx]
        tcr_id = sample['TRB_CDR3']  # Column name for TCR IDs
        epitope_id = sample['Epitope']  # Column name for epitope IDs
        label = sample['Binding']  # Target label

        # Access embeddings lazily
        tcr_embedding = self.tcr_embeddings[tcr_id][:]
        epitope_embedding = self.epitope_embeddings[epitope_id][:]

        return (
            torch.tensor(tcr_embedding, dtype=torch.float32),
            torch.tensor(epitope_embedding, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )

class TCR_Epitope_Transformer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_layers=4, 
                  dropout=0.1):
        super(TCR_Epitope_Transformer, self).__init__()
        # Input embeddings
        self.tcr_embedding = nn.Linear(1024, embed_dim)
        self.epitope_embedding = nn.Linear(1024, embed_dim)
        
        # Positional encodings (now dynamically sized)
        self.tcr_pos = nn.Parameter(torch.randn(1, max_tcr_length, embed_dim))  # Max length 1000
        self.epitope_pos = nn.Parameter(torch.randn(1, max_epitope_length, embed_dim))
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        # Output
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),  # Combine TCR+epitope
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, tcr, epitope):
        # Get masks
        tcr_mask = (tcr.sum(dim=-1) == 0)
        epitope_mask = (epitope.sum(dim=-1) == 0)
        
        # Embed and add positional encoding
        tcr = self.tcr_embedding(tcr) + self.tcr_pos[:, :tcr.size(1), :]
        epitope = self.epitope_embedding(epitope) + self.epitope_pos[:, :epitope.size(1), :]
        
        # Cross-attention layers
        for layer in self.cross_attn_layers:
            tcr, epitope = layer(tcr, epitope, tcr_mask, epitope_mask)
        
        # Pooling
        tcr_pooled = tcr.mean(dim=1)  # (batch, embed_dim)
        epitope_pooled = epitope.mean(dim=1)
        
        # Combine features
        combined = torch.cat([tcr_pooled, epitope_pooled], dim=-1)
        return self.output_layer(combined).squeeze(-1)