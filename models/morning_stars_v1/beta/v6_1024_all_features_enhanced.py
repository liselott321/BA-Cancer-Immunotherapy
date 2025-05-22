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

class FeatureStratifiedSampler:
    """Batch sampler that stratifies by binding status, MHC class, and TCR/epitope properties."""
    
    def __init__(self, dataset, dataframe, batch_size=32):
        self.dataset = dataset
        self.dataframe = dataframe
        self.batch_size = batch_size
        
        # Ensure we have the necessary feature columns
        if "MHC_Class" not in dataframe.columns:
            print("Adding MHC_Class column...")
            def determine_mhc_class(mhc):
                if pd.isna(mhc):
                    return "Unknown"
                elif mhc.startswith(("HLA-A", "HLA-B", "HLA-C")):
                    return "MHC-I"
                elif mhc.startswith("HLA-D"):
                    return "MHC-II"
                else:
                    return "Other"
            dataframe["MHC_Class"] = dataframe["MHC"].apply(determine_mhc_class)
        
        if "TCR_Category" not in dataframe.columns:
            print("Adding TCR_Category column...")
            def categorize_tcr(tcr):
                if pd.isna(tcr) or len(tcr) == 0:
                    return "Unknown"
                length_cat = "Short" if len(tcr) < 12 else "Medium" if len(tcr) < 16 else "Long"
                motif = tcr[:3] if len(tcr) >= 3 else tcr
                return f"{motif}_{length_cat}"
            dataframe["TCR_Category"] = dataframe["TRB_CDR3"].apply(categorize_tcr)
        
        # Create stratification based on binding, MHC class, and TCR category
        self.strata = {}
        binding_values = dataframe["Binding"].unique()
        mhc_classes = dataframe["MHC_Class"].unique()
        
        # Get top TCR categories that have at least 10 examples
        tcr_counts = dataframe["TCR_Category"].value_counts()
        tcr_categories = tcr_counts[tcr_counts >= 10].index.tolist()
        
        # Create strata
        for binding in binding_values:
            for mhc_class in mhc_classes:
                # For TCR categories, we'll either use the specific category or "Other"
                for tcr_cat in tcr_categories + ["Other"]:
                    if tcr_cat == "Other":
                        # "Other" includes all TCR categories not in our main list
                        mask = (
                            (dataframe["Binding"] == binding) & 
                            (dataframe["MHC_Class"] == mhc_class) & 
                            (~dataframe["TCR_Category"].isin(tcr_categories))
                        )
                    else:
                        mask = (
                            (dataframe["Binding"] == binding) & 
                            (dataframe["MHC_Class"] == mhc_class) & 
                            (dataframe["TCR_Category"] == tcr_cat)
                        )
                    
                    indices = np.where(mask)[0]
                    if len(indices) > 0:
                        self.strata[(binding, mhc_class, tcr_cat)] = indices
                        binding_str = "Binding" if binding == 1 else "Non-binding"
                        print(f"Stratum ({binding_str}, {mhc_class}, {tcr_cat}): {len(indices)} examples")
        
        # Initialize pointers and shuffle each stratum
        self.pointers = {k: 0 for k in self.strata}
        for k in self.strata:
            np.random.shuffle(self.strata[k])
    
    def get_batch_indices(self):
        """Get balanced indices for a batch."""
        batch_indices = []
        strata_keys = list(self.strata.keys())
        
        # How many samples we should take from each stratum
        samples_per_stratum = max(1, self.batch_size // len(strata_keys))
        
        # Collect indices from each stratum
        for key in strata_keys:
            indices = self.strata[key]
            pointer = self.pointers[key]
            
            # If we've used all indices in this stratum, shuffle and reset
            if pointer >= len(indices):
                np.random.shuffle(indices)
                self.pointers[key] = 0
                pointer = 0
            
            # How many samples we can take from this stratum
            available = min(samples_per_stratum, len(indices) - pointer)
            
            if available > 0:
                selected = indices[pointer:pointer+available]
                self.pointers[key] += available
                batch_indices.extend(selected)
        
        # If we couldn't fill the batch with our strategy, add random samples
        if len(batch_indices) < self.batch_size:
            remaining = self.batch_size - len(batch_indices)
            all_indices = np.arange(len(self.dataframe))
            remaining_indices = np.setdiff1d(all_indices, batch_indices)
            
            if len(remaining_indices) >= remaining:
                extra = np.random.choice(remaining_indices, size=remaining, replace=False)
            else:
                extra = np.random.choice(all_indices, size=remaining, replace=True)
                
            batch_indices.extend(extra)
        
        # Shuffle the batch indices
        np.random.shuffle(batch_indices)
        return batch_indices
    
    def get_loader(self):
        """Get a DataLoader for a single balanced batch."""
        indices = self.get_batch_indices()
        subset = Subset(self.dataset, indices)
        return DataLoader(subset, batch_size=self.batch_size, shuffle=False)

        
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