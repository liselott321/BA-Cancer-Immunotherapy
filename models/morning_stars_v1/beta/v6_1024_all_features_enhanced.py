import torch
import torch.nn as nn
import numpy as np
from torch.utils.checkpoint import checkpoint

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

class BidirectionalCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        # TCR attends to epitope
        self.tcr_to_epi_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.tcr_norm = nn.LayerNorm(embed_dim)
        self.tcr_dropout = nn.Dropout(dropout)
        
        # Epitope attends to TCR
        self.epi_to_tcr_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.epi_norm = nn.LayerNorm(embed_dim)
        self.epi_dropout = nn.Dropout(dropout)
    
    def forward(self, tcr_emb, epitope_emb, tcr_mask=None, epitope_mask=None):
        # TCR attends to epitope
        tcr_attn_out, _ = self.tcr_to_epi_attn(
            tcr_emb,      # query 
            epitope_emb,  # key
            epitope_emb,  # value
            key_padding_mask=epitope_mask
        )
        tcr_updated = self.tcr_norm(tcr_emb + self.tcr_dropout(tcr_attn_out))
        
        # Epitope attends to TCR
        epi_attn_out, _ = self.epi_to_tcr_attn(
            epitope_emb,  # query
            tcr_emb,      # key
            tcr_emb,      # value
            key_padding_mask=tcr_mask
        )
        epi_updated = self.epi_norm(epitope_emb + self.epi_dropout(epi_attn_out))
        
        return tcr_updated, epi_updated

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
        # tcr_physchem = self.physchem_data["tcr_encoded"][phys_idx]
        # epi_physchem = self.physchem_data["epi_encoded"][phys_idx]
        tcr_physchem = self.physchem_data["tcr_encoded"][int(phys_idx)]  # Convert to int   
        epi_physchem = self.physchem_data["epi_encoded"][int(phys_idx)]
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


class TCR_Epitope_Transformer_Enhanced(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=2,
                 max_tcr_length=43, max_epitope_length=43, dropout=0.1,
                 classifier_hidden_dim=64, physchem_dim=10,
                 trbv_vocab_size=50, trbj_vocab_size=20, mhc_vocab_size=100,
                 use_checkpointing=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_checkpointing = use_checkpointing
        
        # Embeddings
        self.tcr_embedding = nn.Linear(1024, embed_dim)
        self.epitope_embedding = nn.Linear(1024, embed_dim)

        # Categorical feature embeddings
        self.trbv_embed = nn.Embedding(trbv_vocab_size, embed_dim, padding_idx=trbv_vocab_size - 1)
        self.trbj_embed = nn.Embedding(trbj_vocab_size, embed_dim, padding_idx=trbj_vocab_size - 1)
        self.mhc_embed = nn.Embedding(mhc_vocab_size, embed_dim, padding_idx=mhc_vocab_size - 1)

        # Positional encodings
        self.tcr_positional_encoding = nn.Parameter(torch.randn(1, max_tcr_length, embed_dim))
        self.epitope_positional_encoding = nn.Parameter(torch.randn(1, max_epitope_length, embed_dim))

        # Self-attention transformer layers for TCR
        self.transformer_layers = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        
        # Enhanced bidirectional cross-attention
        self.cross_attn_block = BidirectionalCrossAttention(embed_dim, num_heads, dropout)

        # Physicochemical embeddings
        self.tcr_physchem_embed = PeriodicEmbedding(physchem_dim)
        self.epi_physchem_embed = PeriodicEmbedding(physchem_dim)

        # Classifier input dimension calculation
        self.classifier_input_dim = embed_dim * 5 + self.tcr_physchem_embed.output_dim * 2
        self.classifier = Classifier(self.classifier_input_dim, classifier_hidden_dim, dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights for better training stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
    
    def _run_attention_block(self, layer, x, mask=None):
        """Helper function for checkpointing"""
        return layer(x, key_padding_mask=mask)
        
    def forward(self, tcr, epitope, tcr_physchem, epi_physchem, trbv, trbj, mhc):
        # Embed sequences
        tcr_emb = self.tcr_embedding(tcr)
        epitope_emb = self.epitope_embedding(epitope)

        # Embed categorical features
        trbv_emb = self.trbv_embed(trbv).squeeze(1)
        trbj_emb = self.trbj_embed(trbj).squeeze(1)
        mhc_emb = self.mhc_embed(mhc).squeeze(1)

        # Padding masks (True where padding exists)
        tcr_mask = (tcr.sum(dim=-1) == 0)
        epitope_mask = (epitope.sum(dim=-1) == 0)

        # Add positional encodings
        tcr_emb += self.tcr_positional_encoding[:, :tcr_emb.size(1), :]
        epitope_emb += self.epitope_positional_encoding[:, :epitope_emb.size(1), :]

        # Process TCR through transformer layers with gradient checkpointing if enabled
        for i, layer in enumerate(self.transformer_layers):
            if self.use_checkpointing and self.training:
                tcr_emb = checkpoint(self._run_attention_block, layer, tcr_emb, tcr_mask)
            else:
                tcr_emb = layer(tcr_emb, key_padding_mask=tcr_mask)

        # Enhanced bidirectional cross-attention between TCR and epitope
        tcr_updated, epitope_updated = self.cross_attn_block(tcr_emb, epitope_emb, 
                                                           tcr_mask=tcr_mask, 
                                                           epitope_mask=epitope_mask)

        # Masked mean pooling for TCR
        tcr_emb_masked = tcr_updated.masked_fill(tcr_mask.unsqueeze(-1), 0)
        tcr_lengths = (~tcr_mask).sum(dim=1, keepdim=True).clamp(min=1)
        tcr_pooled = tcr_emb_masked.sum(dim=1) / tcr_lengths
        
        # Masked mean pooling for epitope
        epitope_emb_masked = epitope_updated.masked_fill(epitope_mask.unsqueeze(-1), 0)
        epitope_lengths = (~epitope_mask).sum(dim=1, keepdim=True).clamp(min=1)
        epitope_pooled = epitope_emb_masked.sum(dim=1) / epitope_lengths

        # Embed physicochemical properties
        tcr_phys = self.tcr_physchem_embed(tcr_physchem)
        epi_phys = self.epi_physchem_embed(epi_physchem)

        # Concatenate all features
        final_vector = torch.cat([
            tcr_pooled, epitope_pooled,
            trbv_emb, trbj_emb, mhc_emb,
            tcr_phys, epi_phys
        ], dim=1)

        # Final classification
        return self.classifier(final_vector).squeeze(1)