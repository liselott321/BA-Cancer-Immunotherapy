import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
    def __init__(self, data, tcr_embeddings, epitope_embeddings):
        self.data = data
        self.tcr_embeddings = tcr_embeddings
        self.epitope_embeddings = epitope_embeddings

        self.trbv_dict = {val: idx for idx, val in enumerate(data['TRBV'].unique())}
        self.trbj_dict = {val: idx for idx, val in enumerate(data['TRBJ'].unique())}
        self.mhc_dict = {val: idx for idx, val in enumerate(data['MHC'].unique())}

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tcr_embedding = torch.tensor(self.tcr_embeddings[row['TRB_CDR3']], dtype=torch.float32)
        epitope_embedding = torch.tensor(self.epitope_embeddings[row['Epitope']], dtype=torch.float32)
        label = torch.tensor(row['Binding'], dtype=torch.float32)

        trbv = self.trbv_dict[row['TRBV']]
        trbj = self.trbj_dict[row['TRBJ']]
        mhc = self.mhc_dict[row['MHC']]
        task = row['task']
    
        return tcr_embedding, epitope_embedding, label, trbv, trbj, mhc, task

class TCR_Epitope_Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, max_tcr_length, max_epitope_length,
                 num_trbv, num_trbj, num_mhc, dropout=0.1):
        super().__init__()

        self.tcr_embedding = nn.Linear(512, embed_dim)
        self.epitope_embedding = nn.Linear(512, embed_dim)
        self.tcr_positional_encoding = nn.Parameter(torch.randn(1, max_tcr_length, embed_dim))
        self.epitope_positional_encoding = nn.Parameter(torch.randn(1, max_epitope_length, embed_dim))

        self.trbv_embed = nn.Embedding(num_trbv, embed_dim)
        self.trbj_embed = nn.Embedding(num_trbj, embed_dim)
        self.mhc_embed = nn.Embedding(num_mhc, embed_dim)

        # Transformer
        self.transformer_layers = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(embed_dim, 1)

    def forward(self, tcr, epitope, trbv, trbj, mhc):
        tcr = self.tcr_embedding(tcr) + self.tcr_positional_encoding
        epitope = self.epitope_embedding(epitope) + self.epitope_positional_encoding

        # Embed categorical inputs
        trbv_embed = self.trbv_embed(trbv).unsqueeze(1)  # (B, 1, D)
        trbj_embed = self.trbj_embed(trbj).unsqueeze(1)
        mhc_embed = self.mhc_embed(mhc).unsqueeze(1)

        # Combine everything as one sequence
        combined = torch.cat((tcr, epitope, trbv_embed, trbj_embed, mhc_embed), dim=1)

        for layer in self.transformer_layers:
            combined = layer(combined)

        pooled = combined.mean(dim=1)
        output = self.output_layer(pooled).squeeze(1)
        return output