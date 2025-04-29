import torch
import torch.nn as nn
import h5py
import math
import numpy as np

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

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_output, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tcr_emb, epitope_emb):
        # epitope_emb: [B, L_e, D] → als Query
        # tcr_emb: [B, L_t, D] → als Key/Value
        attn_out, _ = self.cross_attn(epitope_emb, tcr_emb, tcr_emb)
        return self.norm(epitope_emb + self.dropout(attn_out))  # residual

class PeriodicEmbedding(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim * 2

    def forward(self, x):
        # Erwartet (batch_size, input_dim)
        div_term = torch.exp(
            torch.arange(0, self.input_dim, device=x.device, dtype=x.dtype) *
            -(np.log(10000.0) / self.input_dim)
        )  # (input_dim,)

        pe = torch.cat([torch.sin(x * div_term), torch.cos(x * div_term)], dim=-1)
        return pe

class LazyTCR_Epitope_Descriptor_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_frame, tcr_embeddings, epitope_embeddings, physchem_h5_path):
        self.data_frame = data_frame
        self.tcr_embeddings = tcr_embeddings
        self.epitope_embeddings = epitope_embeddings
        self.physchem_data = physchem_h5_path

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sample = self.data_frame.iloc[idx]
        tcr_id = sample["TRB_CDR3"]
        epitope_id = sample["Epitope"]
        label = sample["Binding"]
        phys_idx = sample["physchem_index"]  # <- RICHTIG
    
        tcr_embedding = self.tcr_embeddings[tcr_id][:]
        epitope_embedding = self.epitope_embeddings[epitope_id][:]
        tcr_physchem = self.physchem_data["tcr_encoded"][phys_idx]
        epi_physchem = self.physchem_data["epi_encoded"][phys_idx]
    
        return (
            torch.tensor(tcr_embedding, dtype=torch.float32),
            torch.tensor(epitope_embedding, dtype=torch.float32),
            torch.tensor(tcr_physchem, dtype=torch.float32),
            torch.tensor(epi_physchem, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )

class TCR_Epitope_Transformer(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=2, max_tcr_length=43, max_epitope_length=43,
                 dropout=0.1, classifier_hidden_dim=64, physchem_dim=10):
        super(TCR_Epitope_Transformer, self).__init__()

        self.embed_dim = embed_dim
        self.tcr_embedding = nn.Linear(1024, embed_dim)
        self.epitope_embedding = nn.Linear(1024, embed_dim)

        self.tcr_positional_encoding = nn.Parameter(torch.randn(1, max_tcr_length, embed_dim))
        self.epitope_positional_encoding = nn.Parameter(torch.randn(1, max_epitope_length, embed_dim))

        self.transformer_layers = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.cross_attn_block = CrossAttentionBlock(embed_dim, num_heads, dropout)

        self.physchem_dim = physchem_dim
        self.tcr_physchem_embed = PeriodicEmbedding(physchem_dim)
        self.epi_physchem_embed = PeriodicEmbedding(physchem_dim)

        self.classifier_input_dim = embed_dim * 2 + self.tcr_physchem_embed.output_dim * 2
        self.classifier = Classifier(self.classifier_input_dim, classifier_hidden_dim, dropout)

    def forward(self, tcr, epitope, tcr_physchem=None, epi_physchem=None):
        tcr_emb = self.tcr_embedding(tcr)
        epitope_emb = self.epitope_embedding(epitope)
    
        # Create masks
        tcr_mask = (tcr.sum(dim=-1) == 0)
        epitope_mask = (epitope.sum(dim=-1) == 0)
    
        # Add positional encoding
        tcr_emb += self.tcr_positional_encoding[:, :tcr_emb.size(1), :]
        epitope_emb += self.epitope_positional_encoding[:, :epitope_emb.size(1), :]
    
        # Normaler TCR-Epitope Transformer
        for layer in self.transformer_layers:
            tcr_emb = layer(tcr_emb)  # Self-Attention nur auf TCR
        
        # Epitope Conditioning
        cond_out = self.cross_attn_block(tcr_emb, epitope_emb)  # Cross-Attention Epitope ← TCR
        
        # Pooling
        tcr_flat = tcr_emb.mean(dim=1)
        epi_cond = cond_out.mean(dim=1)
        
        # Physicochem + concat
        if tcr_physchem is not None and epi_physchem is not None:
            tcr_phys = self.tcr_physchem_embed(tcr_physchem)
            epi_phys = self.epi_physchem_embed(epi_physchem)
            final_vector = torch.cat([tcr_flat, epi_cond, tcr_phys, epi_phys], dim=1)
        else:
            final_vector = torch.cat([tcr_flat, epi_cond], dim=1)
        
        output = self.classifier(final_vector).squeeze(1)