import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),         # Normalize inputs
            nn.ReLU(),                          # Non-linearity
            nn.Linear(hidden_dim, hidden_dim),  # Fully connected layer
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), 
            nn.Dropout(dropout),                # Dropout for regularization
            nn.Linear(hidden_dim, hidden_dim)   # Another linear layer
        )

    def forward(self, x):
        # Add residual connection
        return x + self.block(x)


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            ResidualBlock(input_dim, dropout),
            nn.Linear(input_dim, 1)              # Final binary classification output
        )

    def forward(self, x):
        return self.model(x)

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # Self-attention + residual + norm
        attn_output, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feedforward + residual + norm
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x

class LazyTCR_Epitope_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_frame, tcr_embeddings, epitope_embeddings,
                 trbv_dict, trbj_dict, mhc_dict):
        self.data_frame = data_frame
        self.tcr_embeddings = tcr_embeddings
        self.epitope_embeddings = epitope_embeddings

        # Mapping from TRBV/TRBJ/MHC strings to indices
        self.trbv_dict = trbv_dict
        self.trbj_dict = trbj_dict
        self.mhc_dict = mhc_dict

        # Index used for unknown elements
        self.unknown_trbv_idx = len(trbv_dict)
        self.unknown_trbj_idx = len(trbj_dict)
        self.unknown_mhc_idx = len(mhc_dict)

        # Map each categorical feature to an index (unknown → fallback index)
        self.data_frame["TRBV_Index"] = data_frame["TRBV"].map(trbv_dict).fillna(self.unknown_trbv_idx).astype(int)
        self.data_frame["TRBJ_Index"] = data_frame["TRBJ"].map(trbj_dict).fillna(self.unknown_trbj_idx).astype(int)
        self.data_frame["MHC_Index"] = data_frame["MHC"].map(mhc_dict).fillna(self.unknown_mhc_idx).astype(int)

        # Safety check
        assert self.data_frame["TRBV_Index"].max() < (self.unknown_trbv_idx + 1), "TRBV_Index out of range!"
        assert self.data_frame["TRBJ_Index"].max() < (self.unknown_trbj_idx + 1), "TRBJ_Index out of range!"
        assert self.data_frame["MHC_Index"].max() < (self.unknown_mhc_idx + 1), "MHC_Index out of range!"


    def __getitem__(self, idx):
        sample = self.data_frame.iloc[idx]
        tcr_id = sample["TRB_CDR3"]
        epitope_id = sample["Epitope"]
        label = sample["Binding"]

        # Get embedding vectors for TCR and epitope by ID
        tcr_embedding = self.tcr_embeddings[tcr_id][:]
        epitope_embedding = self.epitope_embeddings[epitope_id][:]

        return (
            torch.tensor(tcr_embedding, dtype=torch.float32),
            torch.tensor(epitope_embedding, dtype=torch.float32),
            torch.tensor(sample["TRBV_Index"], dtype=torch.long),
            torch.tensor(sample["TRBJ_Index"], dtype=torch.long),
            torch.tensor(sample["MHC_Index"], dtype=torch.long),
            torch.tensor(label, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.data_frame)


class TCR_Epitope_Transformer(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=2, max_tcr_length=20, max_epitope_length=15,
             dropout=0.1, classifier_hidden_dim=64,
             trbv_vocab_size=50, trbj_vocab_size=20, mhc_vocab_size=100):
        super(TCR_Epitope_Transformer, self).__init__()
        self.embed_dim = embed_dim
        
        # Project TCR and Epitope embeddings to model dimension
        self.tcr_embedding = nn.Linear(1024, embed_dim)
        self.epitope_embedding = nn.Linear(1024, embed_dim)
        
        # Embeddings for categorical variables
        self.trbv_embed = nn.Embedding(trbv_vocab_size, embed_dim, padding_idx=trbv_vocab_size - 1)
        self.trbj_embed = nn.Embedding(trbj_vocab_size, embed_dim, padding_idx=trbj_vocab_size - 1)
        self.mhc_embed  = nn.Embedding(mhc_vocab_size,  embed_dim, padding_idx=mhc_vocab_size - 1)
        
        # Positional encodings for sequence order information
        self.tcr_positional_encoding = nn.Parameter(torch.randn(1, max_tcr_length, embed_dim))
        self.epitope_positional_encoding = nn.Parameter(torch.randn(1, max_epitope_length, embed_dim))

        # Stack of transformer attention layers
        self.transformer_layers = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])

        # Input size = sequence output + additional categorical embeddings
        self.classifier_input_dim = (max_tcr_length + max_epitope_length) * embed_dim + 3 * embed_dim
        self.classifier = Classifier(self.classifier_input_dim, classifier_hidden_dim, dropout)


    def forward(self, tcr, epitope, trbv, trbj, mhc):
        # Project embeddings
        tcr_emb = self.tcr_embedding(tcr)
        epitope_emb = self.epitope_embedding(epitope)

        # Embed categorical features
        trbv_embed = self.trbv_embed(trbv).squeeze(1)
        trbj_embed = self.trbj_embed(trbj).squeeze(1)
        mhc_embed = self.mhc_embed(mhc).squeeze(1)

        # Create padding masks for attention
        tcr_mask = (tcr.sum(dim=-1) == 0)
        epitope_mask = (epitope.sum(dim=-1) == 0)

        # Add positional encoding to token embeddings
        tcr_emb += self.tcr_positional_encoding[:, :tcr_emb.size(1), :]
        epitope_emb += self.epitope_positional_encoding[:, :epitope_emb.size(1), :]

        # Concatenate TCR and Epitope sequences
        combined = torch.cat([tcr_emb, epitope_emb], dim=1)
        key_padding_mask = torch.cat([tcr_mask, epitope_mask], dim=1)

        # Pass through transformer layers
        for layer in self.transformer_layers:
            combined = layer(combined, key_padding_mask=key_padding_mask)

        # Flatten the sequence output
        flattened = combined.view(combined.size(0), -1)

        # Concatenate with categorical feature embeddings
        combined_features = torch.cat([flattened, trbv_embed, trbj_embed, mhc_embed], dim=1)

        # Final classification
        output = self.classifier(combined_features).squeeze(1)
        return output