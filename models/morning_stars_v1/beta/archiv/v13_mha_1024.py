import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)  # Pre-LN before attention
        self.dropout1 = nn.Dropout(dropout)

        # Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

        self.norm2 = nn.LayerNorm(embed_dim)  # Pre-LN before FFN
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # Pre-LN before attention
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(
            x_norm.permute(1, 0, 2),
            x_norm.permute(1, 0, 2),
            x_norm.permute(1, 0, 2),
            key_padding_mask=key_padding_mask
        )
        x = x + self.dropout1(attn_output.permute(1, 0, 2))  # Residual connection

        # Pre-LN before FFN
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout2(ffn_output)  # Residual connection

        return x

class TCR_Epitope_Dataset(Dataset):
    def __init__(self, data, tcr_embeddings, epitope_embeddings):
        self.data = data
        self.tcr_embeddings = tcr_embeddings
        self.epitope_embeddings = epitope_embeddings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tcr_embedding = torch.tensor(self.tcr_embeddings[row['TRB_CDR3']], dtype=torch.float32)
        epitope_embedding = torch.tensor(self.epitope_embeddings[row['Epitope']], dtype=torch.float32)
        label = torch.tensor(row['Binding'], dtype=torch.float32)  
        return tcr_embedding, epitope_embedding, label


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
    def __init__(self, embed_dim, num_heads, num_layers, max_tcr_length, max_epitope_length, dropout=0.1):
        super(TCR_Epitope_Transformer, self).__init__()
        self.tcr_embedding = nn.Linear(1024, embed_dim)
        self.epitope_embedding = nn.Linear(1024, embed_dim)

        self.tcr_positional_encoding = nn.Parameter(torch.randn(1, max_tcr_length, embed_dim))
        self.epitope_positional_encoding = nn.Parameter(torch.randn(1, max_epitope_length, embed_dim))

        self.transformer_layers = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(embed_dim, 1)

    def forward(self, tcr, epitope):
        # Compute embeddings
        tcr_embedded = self.tcr_embedding(tcr)
        epitope_embedded = self.epitope_embedding(epitope)

        # Create key padding mask BEFORE adding positional encodings
        tcr_mask = (tcr.sum(dim=-1) == 0)  # Mask positions where TCR values are zero
        epitope_mask = (epitope.sum(dim=-1) == 0)  # Mask positions where epitope values are zero

        # Add positional encodings
        tcr = tcr_embedded + self.tcr_positional_encoding[:, :tcr_embedded.size(1), :]
        epitope = epitope_embedded + self.epitope_positional_encoding[:, :epitope_embedded.size(1), :]

        # Concatenate TCR and epitope along the sequence length dimension
        combined = torch.cat((tcr, epitope), dim=1)
        key_padding_mask = torch.cat((tcr_mask, epitope_mask), dim=1)  # Combine TCR and epitope masks

        # Pass through transformer layers with masking
        for layer in self.transformer_layers:
            combined = layer(combined, key_padding_mask=key_padding_mask)

        # Pooling and output layer
        pooled = combined.mean(dim=1)  # Mean pooling across sequence
        output = self.output_layer(pooled).squeeze(1)

        return output
