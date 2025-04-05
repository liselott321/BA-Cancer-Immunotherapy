import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
    def __init__(self, df, tcr_emb, epi_emb, trbv_dict, trbj_dict, mhc_dict, descriptor_file):
        self.data_frame = df
        self.tcr_emb = tcr_emb
        self.epi_emb = epi_emb
        self.trbv_dict = trbv_dict
        self.trbj_dict = trbj_dict
        self.mhc_dict = mhc_dict
        self.desc_data = h5py.File(descriptor_file, 'r')

    def __len__(self):
        return len(self.data_frame)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tcr_embed = self.tcr_emb[row["TRB_CDR3"]][:]
        epi_embed = self.epi_emb[row["Epitope"]][:]
        tcr_desc = self.desc_data["tcr_encoded"][idx]
        epi_desc = self.desc_data["epi_encoded"][idx]
        label = row['Binding']
        task = row['task']
    
        trbv = self.trbv_dict.get(row['TRBV'], 0)
        trbj = self.trbj_dict.get(row['TRBJ'], 0)
        mhc = self.mhc_dict.get(row['MHC'], 0)

        if row["TRBV"] not in self.trbv_dict:
            print(f"Fehlender TRBV: {row['TRBV']}")
    
        return (
            torch.tensor(tcr_embed, dtype=torch.float32),
            torch.tensor(epi_embed, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(trbv, dtype=torch.long),
            torch.tensor(trbj, dtype=torch.long),
            torch.tensor(mhc, dtype=torch.long),
            task
        )
        
class TCR_Epitope_Transformer_WithDescriptors(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, tcr_descriptor_dim, epi_descriptor_dim,
                 num_trbv, num_trbj, num_mhc, num_tasks, dropout=0.1):
        super().__init__()

        # Projection
        self.protbert_tcr = nn.Sequential(nn.Linear(1024, embed_dim), nn.Tanh())
        self.protbert_epi = nn.Sequential(nn.Linear(1024, embed_dim), nn.Tanh())
        self.desc_tcr = nn.Sequential(nn.Linear(tcr_descriptor_dim, embed_dim), nn.Tanh())
        self.desc_epi = nn.Sequential(nn.Linear(epi_descriptor_dim, embed_dim), nn.Tanh())

        self.trbv_embedding = nn.Embedding(num_trbv, embed_dim)
        self.trbj_embedding = nn.Embedding(num_trbj, embed_dim)
        self.mhc_embedding = nn.Embedding(num_mhc, embed_dim)
        self.task_embedding = nn.Embedding(num_tasks, embed_dim)


        # Feature Gates
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

        # Bottleneck Layer
        self.bottleneck = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Attention Pooling Layer (statt mean pooling)
        self.attn_pool = nn.Linear(embed_dim, 1)

        # Output Layer
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, tcr_embed, epi_embed, tcr_desc, epi_desc, trbv, trbj, mhc):
        # 1. Embedding-Projektionen
        tcr_seq = self.protbert_tcr(tcr_embed)  # [B, L, D]
        epi_seq = self.protbert_epi(epi_embed)  # [B, L, D]

        tcr_desc_proj = self.desc_tcr(tcr_desc).unsqueeze(1).repeat(1, tcr_seq.shape[1], 1)
        epi_desc_proj = self.desc_epi(epi_desc).unsqueeze(1).repeat(1, epi_seq.shape[1], 1)

        # 2. Gating
        tcr_concat = torch.cat([tcr_seq, tcr_desc_proj], dim=-1)
        epi_concat = torch.cat([epi_seq, epi_desc_proj], dim=-1)

        tcr_gate = self.tcr_gate(tcr_concat)
        epi_gate = self.epi_gate(epi_concat)

        tcr = tcr_seq * tcr_gate + tcr_desc_proj * (1 - tcr_gate)
        epi = epi_seq * epi_gate + epi_desc_proj * (1 - epi_gate)

        # Lookup: TRBV, TRBJ, MHC (shape: [batch_size, embed_dim])
        trbv_embed = self.trbv_embedding(trbv).unsqueeze(1)
        trbj_embed = self.trbj_embedding(trbj).unsqueeze(1)
        mhc_embed = self.mhc_embedding(mhc).unsqueeze(1)
        task_embed = self.task_embedding(task).unsqueeze(1)

        # 3. Combine & Attention
        combined = torch.cat((tcr, epi, trbv_embed, trbj_embed, mhc_embed, task_embed), dim=1)
        mask = (combined.sum(dim=-1) == 0)

        for layer in self.transformer_layers:
            combined = layer(combined, key_padding_mask=mask)

        # 4. Bottleneck
        # Attention-Pooling
        weights = torch.softmax(self.attn_pool(combined), dim=1)  # [B, seq_len, 1]
        pooled = torch.sum(combined * weights, dim=1)  # [B, D]
        bottlenecked = self.bottleneck(pooled)

        return self.classifier(bottlenecked).squeeze(1)