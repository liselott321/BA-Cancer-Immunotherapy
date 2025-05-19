import torch
import torch.nn as nn
import h5py
import math
import numpy as np
import torch, torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return x + self.block(x)

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dim, dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)

class ReciprocalAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        # TCR → Epitope
        self.cross_t2e = nn.MultiheadAttention(embed_dim, num_heads,
                                               dropout=dropout, batch_first=True)
        # Epitope → TCR
        self.cross_e2t = nn.MultiheadAttention(embed_dim, num_heads,
                                               dropout=dropout, batch_first=True)
        self.norm_t1 = nn.LayerNorm(embed_dim)
        self.norm_e1 = nn.LayerNorm(embed_dim)
        self.ff_t    = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4), nn.ReLU(),
            nn.Dropout(dropout),   nn.Linear(embed_dim*4, embed_dim)
        )
        self.ff_e    = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4), nn.ReLU(),
            nn.Dropout(dropout),   nn.Linear(embed_dim*4, embed_dim)
        )
        self.norm_t2 = nn.LayerNorm(embed_dim)
        self.norm_e2 = nn.LayerNorm(embed_dim)

    def forward(self, tcr, epi, tcr_mask=None, epi_mask=None):
        # TCR attends to Epitope
        t2e, _ = self.cross_t2e(query=tcr, key=epi, value=epi,
                               key_padding_mask=epi_mask)
        tcr = self.norm_t1(tcr + t2e)
        tcr = self.norm_t2(tcr + self.ff_t(tcr))
        # Epitope attends to TCR
        e2t, _ = self.cross_e2t(query=epi, key=tcr, value=tcr,
                               key_padding_mask=tcr_mask)
        epi = self.norm_e1(epi + e2t)
        epi = self.norm_e2(epi + self.ff_e(epi))
        return tcr, epi


class LazyTCR_Epitope_Descriptor_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_frame, tcr_embeddings, epitope_embeddings, ple_h5, ):
        self.data_frame = data_frame
        self.tcr_embeddings = tcr_embeddings
        self.epitope_embeddings = epitope_embeddings
        self.ple_h5    = ple_h5

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sample = self.data_frame.iloc[idx]
        tcr_id = sample["TRB_CDR3"]
        epitope_id = sample["Epitope"]
        label = sample["Binding"]
        phys_idx = int(sample["physchem_index"])
    
        tcr_embedding = self.tcr_embeddings[tcr_id][:]
        epitope_embedding = self.epitope_embeddings[epitope_id][:]
        tcr_ple  = self.ple_h5["tcr_ple"][phys_idx]
        epi_ple  = self.ple_h5["epi_ple"][phys_idx]

    
        return (
            torch.tensor(tcr_embedding, dtype=torch.float32),
            torch.tensor(epitope_embedding, dtype=torch.float32),
            torch.tensor(tcr_ple, dtype=torch.float32),
            torch.tensor(epi_ple, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )

class TCR_Epitope_Transformer(nn.Module):
    def __init__(self, ple_dim, embed_dim=128, num_heads=4, num_layers=2, max_tcr_length=43, max_epitope_length=43,
                 dropout=0.1, classifier_hidden_dim=64):
        super(TCR_Epitope_Transformer, self).__init__()

        self.embed_dim = embed_dim
        self.tcr_embedding = nn.Linear(1024, embed_dim)
        self.epitope_embedding = nn.Linear(1024, embed_dim)
        
        self.tcr_positional_encoding = nn.Parameter(torch.randn(1, max_tcr_length, embed_dim))
        self.epitope_positional_encoding = nn.Parameter(torch.randn(1, max_epitope_length, embed_dim))
        
        self.transformer_layers = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.reciprocal = ReciprocalAttentionBlock(embed_dim, num_heads, dropout)

        seq_dim  = embed_dim * (max_tcr_length + max_epitope_length)
        total_dim = seq_dim + 2 * ple_dim  # tcr_ple + epi_ple

        self.classifier = Classifier(total_dim, classifier_hidden_dim, dropout)


    def forward(self, tcr, epitope, tcr_ple, epi_ple):
        tcr_emb = self.tcr_embedding(tcr)
        epitope_emb = self.epitope_embedding(epitope)
    
        # Add positional encoding
        tcr_emb += self.tcr_positional_encoding[:, :tcr_emb.size(1), :]
        epitope_emb += self.epitope_positional_encoding[:, :epitope_emb.size(1), :]

        # 2) Mask-Variable
        tcr_mask     = (tcr.sum(dim=-1) == 0)
        epitope_mask = (epitope.sum(dim=-1) == 0)
        key_padding_mask = torch.cat([tcr_mask, epitope_mask], dim=1)
    
        # 1) Self-Attention über das gepaddete Kombi-Embedding
        combined = torch.cat([tcr_emb, epitope_emb], dim=1)
        for layer in self.transformer_layers:
            combined = layer(combined, key_padding_mask=key_padding_mask)
            
        tcr_out, epi_out = reciprocal_cross_attention(tcr_emb, epitope_emb, key_padding_mask

        fusion = torch.cat([tcr_out, epi_out], dim=1)          # [B, seq_len*embed_dim]
        seq_feat = fusion.view(fusion.size(0), -1) 

        # --- Fusion & Klassifikation ---
        feat = torch.cat([seq_feat, tcr_ple, epi_ple], dim=1)       
        output = self.classifier(feat).squeeze(-1)                 # [B]
        return output