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
    def __init__(self, data_frame, tcr_embeddings, epitope_embeddings, physchem_h5_path, trbv_dict, trbj_dict, mhc_dict):
        self.data_frame = data_frame
        self.tcr_embeddings = tcr_embeddings
        self.epitope_embeddings = epitope_embeddings
        self.physchem_data = physchem_h5_path
        
        # Dictionary-Zuweisung
        self.trbv_dict = trbv_dict
        self.trbj_dict = trbj_dict
        self.mhc_dict = mhc_dict

        # Compute UNKNOWN-Indices
        self.unknown_trbv_idx = len(trbv_dict)
        self.unknown_trbj_idx = len(trbj_dict)
        self.unknown_mhc_idx = len(mhc_dict)

        self.data_frame["TRBV_Index"] = data_frame["TRBV"].map(trbv_dict).fillna(self.unknown_trbv_idx).astype(int)
        self.data_frame["TRBJ_Index"] = data_frame["TRBJ"].map(trbj_dict).fillna(self.unknown_trbj_idx).astype(int)
        self.data_frame["MHC_Index"] = data_frame["MHC"].map(mhc_dict).fillna(self.unknown_mhc_idx).astype(int)

        # Check 
        assert self.data_frame["TRBV_Index"].max() < (self.unknown_trbv_idx + 1), "TRBV_Index out of range!"
        assert self.data_frame["TRBJ_Index"].max() < (self.unknown_trbj_idx + 1), "TRBJ_Index out of range!"
        assert self.data_frame["MHC_Index"].max() < (self.unknown_mhc_idx + 1), "MHC_Index out of range!"

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
            torch.tensor(sample["TRBV_Index"], dtype=torch.long),
            torch.tensor(sample["TRBJ_Index"], dtype=torch.long),
            torch.tensor(sample["MHC_Index"], dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.data_frame)


class ReciprocalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.peptide_to_tcr = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.tcr_to_peptide = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm_peptide = nn.LayerNorm(embed_dim)
        self.norm_tcr = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tcr_emb, peptide_emb, key_padding_mask_tcr=None, key_padding_mask_epi=None):
        pep2tcr, _ = self.peptide_to_tcr(peptide_emb, tcr_emb, tcr_emb, key_padding_mask=key_padding_mask_tcr)
        peptide_out = self.norm_peptide(peptide_emb + self.dropout(pep2tcr))
    
        tcr2pep, _ = self.tcr_to_peptide(tcr_emb, peptide_emb, peptide_emb, key_padding_mask=key_padding_mask_epi)
        tcr_out = self.norm_tcr(tcr_emb + self.dropout(tcr2pep))
    
        return tcr_out, peptide_out


class TCR_Epitope_Transformer_Reciprocal(nn.Module):
    def __init__(self, trbv_vocab_size, trbj_vocab_size, mhc_vocab_size, embed_dim=128, num_heads=4, max_tcr_length=43, max_epitope_length=43, dropout=0.1, physchem_dim=10, classifier_hidden_dim=64):
        super().__init__()

        self.tcr_embedding = nn.Linear(1024, embed_dim)
        self.epitope_embedding = nn.Linear(1024, embed_dim)
        self.trbv_embedding = nn.Embedding(trbv_vocab_size, embed_dim, padding_idx=-1)
        self.trbj_embedding = nn.Embedding(trbj_vocab_size, embed_dim, padding_idx=-1)
        self.mhc_embedding = nn.Embedding(mhc_vocab_size, embed_dim, padding_idx=-1)
        
        self.tcr_positional_encoding = nn.Parameter(torch.randn(1, max_tcr_length, embed_dim))
        self.epitope_positional_encoding = nn.Parameter(torch.randn(1, max_epitope_length, embed_dim))
        self.rec_attn = ReciprocalAttention(embed_dim, num_heads, dropout)


        self.tcr_physchem_embed = PeriodicEmbedding(physchem_dim)
        self.epi_physchem_embed = PeriodicEmbedding(physchem_dim)

        total_len = max_tcr_length + max_epitope_length
        
        self.classifier_input_dim = (
            embed_dim * (max_tcr_length + max_epitope_length)
            + 2 * self.tcr_physchem_embed.output_dim
            + 3 * embed_dim  # TRBV, TRBJ, MHC
        )
        self.classifier = Classifier(self.classifier_input_dim, classifier_hidden_dim, dropout)

    def forward(self, tcr, epitope, tcr_physchem=None, epi_physchem=None, trbv=None, trbj=None, mhc=None):
        tcr_emb = self.tcr_embedding(tcr) + self.tcr_positional_encoding[:, :tcr.size(1), :]
        epitope_emb = self.epitope_embedding(epitope) + self.epitope_positional_encoding[:, :epitope.size(1), :]

        # Masken berechnen: (batch_size, seq_len)
        tcr_mask = (tcr.sum(dim=-1) == 0)  # True bei Padding
        epi_mask = (epitope.sum(dim=-1) == 0)
        
        # Apply Attention mit Masken
        tcr_out, epitope_out = self.rec_attn(
            tcr_emb, epitope_emb,
            key_padding_mask_tcr=tcr_mask,
            key_padding_mask_epi=epi_mask
        )


        combined = torch.cat([tcr_out, epitope_out], dim=1)
        flattened = combined.view(combined.size(0), -1)

        if tcr_physchem is not None and epi_physchem is not None:
            tcr_physchem = self.tcr_physchem_embed(tcr_physchem)
            epi_physchem = self.epi_physchem_embed(epi_physchem)
            flattened = torch.cat([flattened, tcr_physchem, epi_physchem], dim=1)

        trbv_emb = self.trbv_embedding(trbv)
        trbj_emb = self.trbj_embedding(trbj)
        mhc_emb = self.mhc_embedding(mhc)

        full_input = torch.cat([flattened, trbv_emb, trbj_emb, mhc_emb], dim=1)
        return self.classifier(full_input).squeeze(1)