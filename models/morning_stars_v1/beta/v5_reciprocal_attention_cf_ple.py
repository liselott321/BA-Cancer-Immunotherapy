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

class ReciprocalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        # epitope → TCR
        self.peptide_to_tcr = nn.MultiheadAttention(embed_dim, num_heads,
                                                    dropout=dropout,
                                                    batch_first=True)
        # TCR → epitope
        self.tcr_to_peptide = nn.MultiheadAttention(embed_dim, num_heads,
                                                    dropout=dropout,
                                                    batch_first=True)
        self.norm_peptide = nn.LayerNorm(embed_dim)
        self.norm_tcr     = nn.LayerNorm(embed_dim)
        self.dropout      = nn.Dropout(dropout)

    def forward(self, tcr_emb, epitope_emb,
                key_padding_mask_tcr=None,
                key_padding_mask_epi=None):
        # epitope “fragt” am TCR
        # query = epitope, key/value = tcr
        pep2tcr, _   = self.peptide_to_tcr(epitope_emb, tcr_emb, tcr_emb,
                                           key_padding_mask=key_padding_mask_tcr)
        epitope_out  = self.norm_peptide(epitope_emb + self.dropout(pep2tcr))

        # TCR “fragt” am epitope
        # query = tcr, key/value = epitope
        tcr2pep, _   = self.tcr_to_peptide(tcr_emb, epitope_emb, epitope_emb,
                                           key_padding_mask=key_padding_mask_epi)
        tcr_out      = self.norm_tcr(tcr_emb + self.dropout(tcr2pep))

        return tcr_out, epitope_out



class LazyTCR_Epitope_Descriptor_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_frame, tcr_embeddings, epitope_embeddings, ple_tcr_tensor, ple_epi_tensor, trbv_dict, trbj_dict, mhc_dict):
        self.data_frame = data_frame
        self.tcr_embeddings = tcr_embeddings
        self.epitope_embeddings = epitope_embeddings
        self.ple_tcr_tensor = ple_tcr_tensor
        self.ple_epi_tensor = ple_epi_tensor

        # Dictionary-Zuweisung
        self.trbv_dict = trbv_dict
        self.trbj_dict = trbj_dict
        self.mhc_dict = mhc_dict

        # Berechne UNKNOWN-Indices
        self.unknown_trbv_idx = len(trbv_dict)
        self.unknown_trbj_idx = len(trbj_dict)
        self.unknown_mhc_idx = len(mhc_dict)

        self.data_frame["TRBV_Index"] = data_frame["TRBV"].map(trbv_dict).fillna(self.unknown_trbv_idx).astype(int)
        self.data_frame["TRBJ_Index"] = data_frame["TRBJ"].map(trbj_dict).fillna(self.unknown_trbj_idx).astype(int)
        self.data_frame["MHC_Index"] = data_frame["MHC"].map(mhc_dict).fillna(self.unknown_mhc_idx).astype(int)

        # Sicherheit
        assert self.data_frame["TRBV_Index"].max() < (self.unknown_trbv_idx + 1), "TRBV_Index out of range!"
        assert self.data_frame["TRBJ_Index"].max() < (self.unknown_trbj_idx + 1), "TRBJ_Index out of range!"
        assert self.data_frame["MHC_Index"].max() < (self.unknown_mhc_idx + 1), "MHC_Index out of range!"

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
        tcr_ple = self.ple_tcr_tensor[phys_idx]
        epi_ple = self.ple_epi_tensor[phys_idx]
    
        return (
            torch.tensor(tcr_embedding, dtype=torch.float32),
            torch.tensor(epitope_embedding, dtype=torch.float32),
            torch.as_tensor(tcr_ple, dtype=torch.float32),
            torch.as_tensor(epi_ple, dtype=torch.float32),
            torch.tensor(sample["TRBV_Index"], dtype=torch.long),
            torch.tensor(sample["TRBJ_Index"], dtype=torch.long),
            torch.tensor(sample["MHC_Index"], dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )

class TCR_Epitope_Transformer(nn.Module):
    def __init__(self, trbv_vocab_size, trbj_vocab_size, mhc_vocab_size, ple_dim, embed_dim=128, num_heads=4, num_layers=2, max_tcr_length=43, max_epitope_length=43, dropout=0.1, classifier_hidden_dim=64):
        super(TCR_Epitope_Transformer, self).__init__()

        self.embed_dim = embed_dim
        self.tcr_embedding = nn.Linear(1024, embed_dim)
        self.epitope_embedding = nn.Linear(1024, embed_dim)
        self.ple_reduction = nn.Linear(ple_dim, 64)
        self.trbv_embedding = nn.Embedding(trbv_vocab_size, embed_dim, padding_idx=-1)
        self.trbj_embedding = nn.Embedding(trbj_vocab_size, embed_dim, padding_idx=-1)
        self.mhc_embedding = nn.Embedding(mhc_vocab_size, embed_dim, padding_idx=-1)
        
        self.tcr_positional_encoding = nn.Parameter(torch.randn(1, max_tcr_length, embed_dim))
        self.epitope_positional_encoding = nn.Parameter(torch.randn(1, max_epitope_length, embed_dim))

        # self-attention per sequence
        self.transformer_layers = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # cross-attention
        self.rec_attn = ReciprocalAttention(embed_dim, num_heads, dropout)
        
        seq_dim  = embed_dim * (max_tcr_length + max_epitope_length)
        reduced_ple_dim = 64
        total_dim = seq_dim + 2 * reduced_ple_dim + 3 * embed_dim  # tcr_ple + epi_ple + mhc + tbrj + trbv

        self.classifier = Classifier(total_dim, classifier_hidden_dim, dropout)

    def forward(self, tcr, epitope, tcr_ple, epi_ple, trbv=None, trbj=None, mhc=None):
        tcr_emb = self.tcr_embedding(tcr)
        epitope_emb = self.epitope_embedding(epitope)
        tcr_ple = self.ple_reduction(tcr_ple)
        epi_ple = self.ple_reduction(epi_ple)

        # Embeddings der zusätzlichen Features
        trbv_emb = self.trbv_embedding(trbv)
        trbj_emb = self.trbj_embedding(trbj)
        mhc_emb  = self.mhc_embedding(mhc)
        
        # Create masks
        tcr_mask = (tcr.sum(dim=-1) == 0)
        epitope_mask = (epitope.sum(dim=-1) == 0)
    
        # Add positional encoding
        tcr_emb += self.tcr_positional_encoding[:, :tcr_emb.size(1), :]
        epitope_emb += self.epitope_positional_encoding[:, :epitope_emb.size(1), :]
    
        for layer in self.transformer_layers:
            tcr_emb     = layer(tcr_emb, key_padding_mask=tcr_mask)
            epitope_emb = layer(epitope_emb, key_padding_mask=epitope_mask)

        tcr_out, epitope_out = self.rec_attn(
            tcr_emb, epitope_emb,
            key_padding_mask_tcr=tcr_mask,
            key_padding_mask_epi=epitope_mask
        )
        combined = torch.cat([tcr_out, epitope_out], dim=1)
        seq_feat = combined.view(combined.size(0), -1)
        feat = torch.cat([seq_feat, tcr_ple, epi_ple, trbv_emb, trbj_emb, mhc_emb], dim=1)       
        output = self.classifier(feat).squeeze(-1)                 # [B]
        return output