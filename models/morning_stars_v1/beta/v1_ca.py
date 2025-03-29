import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_output, _ = self.attn(
            x.permute(1, 0, 2),
            x.permute(1, 0, 2),
            x.permute(1, 0, 2),
            key_padding_mask=key_padding_mask
        )
        x = self.norm(x + self.dropout(attn_output.permute(1, 0, 2)))
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_seq, key_value_seq, key_padding_mask=None):
        # Shape: (batch, seq_len, embed_dim) → (seq_len, batch, embed_dim)
        q = query_seq.permute(1, 0, 2)
        kv = key_value_seq.permute(1, 0, 2)

        attn_output, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask)
        out = query_seq + self.dropout(attn_output.permute(1, 0, 2))
        return self.norm(out)


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

class TCR_Epitope_Transformer_Cross(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, max_tcr_length, max_epitope_length, dropout=0.1):
        super(TCR_Epitope_Transformer_Cross, self).__init__()
        self.tcr_embedding = nn.Linear(512, embed_dim)
        self.epitope_embedding = nn.Linear(512, embed_dim)
        
        self.tcr_positional_encoding = nn.Parameter(torch.randn(1, max_tcr_length, embed_dim))
        self.epitope_positional_encoding = nn.Parameter(torch.randn(1, max_epitope_length, embed_dim))

        self.self_attn_blocks = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.cross_attn_tcr_to_epi = CrossAttentionBlock(embed_dim, num_heads, dropout)
        self.cross_attn_epi_to_tcr = CrossAttentionBlock(embed_dim, num_heads, dropout)

        self.output_layer = nn.Linear(embed_dim * 2, 1)  # concat TCR + Epitope pooled

    def forward(self, tcr, epitope):
        # Embedding + Positional Encoding
        tcr = self.tcr_embedding(tcr) + self.tcr_positional_encoding[:, :tcr.size(1), :]
        epitope = self.epitope_embedding(epitope) + self.epitope_positional_encoding[:, :epitope.size(1), :]

        # Self-Attention (optional, kannst du auch rauslassen)
        for layer in self.self_attn_blocks:
            tcr = layer(tcr)
            epitope = layer(epitope)

        # Cross-Attention (TCR attends to Epi und umgekehrt)
        tcr_attn = self.cross_attn_tcr_to_epi(tcr, epitope)
        epi_attn = self.cross_attn_epi_to_tcr(epitope, tcr)

        # Pooling
        pooled_tcr = tcr_attn.mean(dim=1)
        pooled_epi = epi_attn.mean(dim=1)

        # Kombinieren & Vorhersage
        combined = torch.cat((pooled_tcr, pooled_epi), dim=1)
        output = self.output_layer(combined).squeeze(1)
        return output
