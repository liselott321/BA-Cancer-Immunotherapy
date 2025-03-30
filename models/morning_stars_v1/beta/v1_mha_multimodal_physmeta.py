import torch
import torch.nn as nn
from torch.utils.data import Dataset
import h5py

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_output, _ = self.attn(
            x.permute(1, 0, 2),  # Query
            x.permute(1, 0, 2),  # Key
            x.permute(1, 0, 2),  # Value
            key_padding_mask=key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_output.permute(1, 0, 2)))
        return x


class LazyTCR_Epitope_Descriptor_Dataset(Dataset):
    def __init__(self, df, tcr_emb, epi_emb, descriptor_file):
        self.df = df
        self.tcr_emb = tcr_emb
        self.epi_emb = epi_emb
        self.desc_data = h5py.File(descriptor_file, 'r')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tcr_embed = self.tcr_emb[row["TRB_CDR3"]][:]
        epi_embed = self.epi_emb[row["Epitope"]][:]
        tcr_desc = self.desc_data["tcr_encoded"][idx]
        epi_desc = self.desc_data["epi_encoded"][idx]
        label = row["Binding"]

        return (
            torch.tensor(tcr_embed, dtype=torch.float32),
            torch.tensor(epi_embed, dtype=torch.float32),
            torch.tensor(tcr_desc, dtype=torch.float32),
            torch.tensor(epi_desc, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


class TCR_Epitope_Transformer_WithDescriptors(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, tcr_descriptor_dim, epi_descriptor_dim, dropout=0.1):
        super().__init__()

        # Projection
        self.protbert_tcr = nn.Sequential(nn.Linear(1024, embed_dim), nn.Tanh())
        self.protbert_epi = nn.Sequential(nn.Linear(1024, embed_dim), nn.Tanh())
        self.desc_tcr = nn.Sequential(nn.Linear(tcr_descriptor_dim, embed_dim), nn.Tanh())
        self.desc_epi = nn.Sequential(nn.Linear(epi_descriptor_dim, embed_dim), nn.Tanh())

        # Feature Gates (Option 3: dynamische Fusion)
        self.tcr_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        self.epi_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

        # Transformer Layer
        self.transformer_layers = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])

        # Bottleneck Layer (Option 1)
        self.bottleneck = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Output Layer
        self.output_layer = nn.Linear(embed_dim, 1)

    def forward(self, tcr_embed, epi_embed, tcr_desc, epi_desc):
        # 1. Embedding-Projektionen
        tcr_seq = self.protbert_tcr(tcr_embed)  # [B, L, D]
        epi_seq = self.protbert_epi(epi_embed)  # [B, L, D]

        tcr_desc_proj = self.desc_tcr(tcr_desc).unsqueeze(1).repeat(1, tcr_seq.shape[1], 1)
        epi_desc_proj = self.desc_epi(epi_desc).unsqueeze(1).repeat(1, epi_seq.shape[1], 1)

        # 2. Gating (Option 3)
        tcr_concat = torch.cat([tcr_seq, tcr_desc_proj], dim=-1)
        epi_concat = torch.cat([epi_seq, epi_desc_proj], dim=-1)

        tcr_gate = self.tcr_gate(tcr_concat)
        epi_gate = self.epi_gate(epi_concat)

        tcr = tcr_seq * tcr_gate + tcr_desc_proj * (1 - tcr_gate)
        epi = epi_seq * epi_gate + epi_desc_proj * (1 - epi_gate)

        # 3. Combine & Attention
        combined = torch.cat((tcr, epi), dim=1)  # [B, 2L, D]
        mask = (combined.sum(dim=-1) == 0)

        for layer in self.transformer_layers:
            combined = layer(combined, key_padding_mask=mask)

        # 4. Bottleneck (Option 1)
        pooled = combined.mean(dim=1)  # [B, D]
        bottlenecked = self.bottleneck(pooled)

        return self.output_layer(bottlenecked).squeeze(1)
