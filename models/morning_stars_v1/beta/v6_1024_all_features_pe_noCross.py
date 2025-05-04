import torch
import torch.nn as nn
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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
        attn_out, _ = self.cross_attn(epitope_emb, tcr_emb, tcr_emb)
        return self.norm(epitope_emb + self.dropout(attn_out))

class PeriodicEmbedding(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim * 2

    def forward(self, x):
        div_term = torch.exp(
            torch.arange(0, self.input_dim, device=x.device, dtype=x.dtype) *
            -(np.log(10000.0) / self.input_dim)
        )
        pe = torch.cat([torch.sin(x * div_term), torch.cos(x * div_term)], dim=-1)
        return pe


class LazyFullFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, data_frame, tcr_embeddings, epitope_embeddings,
                 physchem_h5_path, trbv_dict, trbj_dict, mhc_dict):
        self.data_frame = data_frame
        self.tcr_embeddings = tcr_embeddings
        self.epitope_embeddings = epitope_embeddings
        self.physchem_data = physchem_h5_path

        # Handle categorical features
        self.trbv_dict = trbv_dict
        self.trbj_dict = trbj_dict
        self.mhc_dict = mhc_dict

        self.unknown_trbv_idx = len(trbv_dict)
        self.unknown_trbj_idx = len(trbj_dict)
        self.unknown_mhc_idx = len(mhc_dict)

        self.data_frame["TRBV_Index"] = data_frame["TRBV"].map(trbv_dict).fillna(self.unknown_trbv_idx).astype(int)
        self.data_frame["TRBJ_Index"] = data_frame["TRBJ"].map(trbj_dict).fillna(self.unknown_trbj_idx).astype(int)
        self.data_frame["MHC_Index"] = data_frame["MHC"].map(mhc_dict).fillna(self.unknown_mhc_idx).astype(int)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sample = self.data_frame.iloc[idx]
        tcr_id = sample["TRB_CDR3"]
        epitope_id = sample["Epitope"]
        label = sample["Binding"]
        phys_idx = sample["physchem_index"]

        tcr_embedding = self.tcr_embeddings[tcr_id][:]
        epitope_embedding = self.epitope_embeddings[epitope_id][:]
        tcr_physchem = self.physchem_data["tcr_encoded"][phys_idx]
        epi_physchem = self.physchem_data["epi_encoded"][phys_idx]

        trbv_index = sample["TRBV_Index"]
        trbj_index = sample["TRBJ_Index"]
        mhc_index = sample["MHC_Index"]

        return (
            torch.tensor(tcr_embedding, dtype=torch.float32),
            torch.tensor(epitope_embedding, dtype=torch.float32),
            torch.tensor(tcr_physchem, dtype=torch.float32),
            torch.tensor(epi_physchem, dtype=torch.float32),
            torch.tensor(trbv_index, dtype=torch.long),
            torch.tensor(trbj_index, dtype=torch.long),
            torch.tensor(mhc_index, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32)
        )



class TCR_Epitope_Transformer_AllFeatures(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=2,
                 max_tcr_length=43, max_epitope_length=43, dropout=0.1,
                 classifier_hidden_dim=64, physchem_dim=10,
                 trbv_vocab_size=50, trbj_vocab_size=20, mhc_vocab_size=100):
        super().__init__()

        self.embed_dim = embed_dim
        self.tcr_embedding = nn.Linear(1024, embed_dim)
        self.epitope_embedding = nn.Linear(1024, embed_dim)

        self.trbv_embed = nn.Embedding(trbv_vocab_size, embed_dim, padding_idx=trbv_vocab_size - 1)
        self.trbj_embed = nn.Embedding(trbj_vocab_size, embed_dim, padding_idx=trbj_vocab_size - 1)
        self.mhc_embed  = nn.Embedding(mhc_vocab_size, embed_dim, padding_idx=mhc_vocab_size - 1)

        self.tcr_positional_encoding = nn.Parameter(torch.randn(1, max_tcr_length, embed_dim))
        self.epitope_positional_encoding = nn.Parameter(torch.randn(1, max_epitope_length, embed_dim))

        # Separate transformer layers for TCR and epitope
        self.tcr_transformer_layers = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.epitope_transformer_layers = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])

        self.tcr_physchem_embed = PeriodicEmbedding(physchem_dim)
        self.epi_physchem_embed = PeriodicEmbedding(physchem_dim)

        self.classifier_input_dim = embed_dim * 5 + self.tcr_physchem_embed.output_dim * 2
        self.classifier = Classifier(self.classifier_input_dim, classifier_hidden_dim, dropout)

    def forward(self, tcr, epitope, tcr_physchem, epi_physchem, trbv, trbj, mhc):
        tcr_emb = self.tcr_embedding(tcr)
        epitope_emb = self.epitope_embedding(epitope)

        trbv_emb = self.trbv_embed(trbv).squeeze(1)
        trbj_emb = self.trbj_embed(trbj).squeeze(1)
        mhc_emb = self.mhc_embed(mhc).squeeze(1)

        # Padding masks
        tcr_mask = (tcr.sum(dim=-1) == 0)
        epitope_mask = (epitope.sum(dim=-1) == 0)

        # Positional encodings
        tcr_emb += self.tcr_positional_encoding[:, :tcr_emb.size(1), :]
        epitope_emb += self.epitope_positional_encoding[:, :epitope_emb.size(1), :]

        # Independent transformer layers for TCR
        for layer in self.tcr_transformer_layers:
            tcr_emb = layer(tcr_emb, key_padding_mask=tcr_mask)

        for layer in self.epitope_transformer_layers:
            epitope_emb = layer(epitope_emb, key_padding_mask=epitope_mask)

        # Masked mean pooling
        tcr_emb_masked = tcr_emb.masked_fill(tcr_mask.unsqueeze(-1), 0)
        epitope_emb_masked = epitope_emb.masked_fill(epitope_mask.unsqueeze(-1), 0)

        tcr_lengths = (~tcr_mask).sum(dim=1, keepdim=True).clamp(min=1)
        epitope_lengths = (~epitope_mask).sum(dim=1, keepdim=True).clamp(min=1)

        tcr_pooled = tcr_emb_masked.sum(dim=1) / tcr_lengths
        epitope_pooled = epitope_emb_masked.sum(dim=1) / epitope_lengths

        # Embed physchem
        tcr_phys = self.tcr_physchem_embed(tcr_physchem)
        epi_phys = self.epi_physchem_embed(epi_physchem)

        # Concatenate all features
        final_vector = torch.cat([
            tcr_pooled, epitope_pooled,
            trbv_emb, trbj_emb, mhc_emb,
            tcr_phys, epi_phys
        ], dim=1)

        return self.classifier(final_vector).squeeze(1)
