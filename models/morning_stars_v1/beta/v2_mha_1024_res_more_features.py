import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
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
            nn.BatchNorm1d(hidden_dim),
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

class LazyTCR_Epitope_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_frame, tcr_embeddings, epitope_embeddings,
                 trbv_dict, trbj_dict, mhc_dict):
        self.data_frame = data_frame
        self.tcr_embeddings = tcr_embeddings
        self.epitope_embeddings = epitope_embeddings

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


    def __getitem__(self, idx):
        sample = self.data_frame.iloc[idx]
        tcr_id = sample["TRB_CDR3"]
        epitope_id = sample["Epitope"]
        label = sample["Binding"]

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
        self.tcr_embedding = nn.Linear(1024, embed_dim)
        self.epitope_embedding = nn.Linear(1024, embed_dim)
        self.trbv_embed = nn.Embedding(trbv_vocab_size, embed_dim, padding_idx=trbv_vocab_size - 1)
        self.trbj_embed = nn.Embedding(trbj_vocab_size, embed_dim, padding_idx=trbj_vocab_size - 1)
        self.mhc_embed  = nn.Embedding(mhc_vocab_size,  embed_dim, padding_idx=mhc_vocab_size - 1)

        self.tcr_bn = nn.BatchNorm1d(max_tcr_length)
        self.epitope_bn = nn.BatchNorm1d(max_epitope_length)

        self.tcr_positional_encoding = nn.Parameter(torch.randn(1, max_tcr_length, embed_dim))
        self.epitope_positional_encoding = nn.Parameter(torch.randn(1, max_epitope_length, embed_dim))

        self.transformer_layers = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])

        #self.classifier_input_dim = embed_dim * 2  # For concatenated mean+max pooling
        # neue Länge beachten:
        self.classifier_input_dim = embed_dim * 5
        self.classifier = Classifier(self.classifier_input_dim, classifier_hidden_dim, dropout)

    def forward(self, tcr, epitope, trbv, trbj, mhc):
        tcr_emb = self.tcr_embedding(tcr)
        epitope_emb = self.epitope_embedding(epitope)
        trbv_embed = self.trbv_embed(trbv).squeeze(1)
        trbj_embed = self.trbj_embed(trbj).squeeze(1)
        mhc_embed = self.mhc_embed(mhc).squeeze(1)

        # Optional: normalize across sequence
        tcr_emb = self.tcr_bn(tcr_emb)
        epitope_emb = self.epitope_bn(epitope_emb)

        # Create masks
        tcr_mask = (tcr.sum(dim=-1) == 0)
        epitope_mask = (epitope.sum(dim=-1) == 0)

        # Add positional encoding
        tcr_emb += self.tcr_positional_encoding[:, :tcr_emb.size(1), :]
        epitope_emb += self.epitope_positional_encoding[:, :epitope_emb.size(1), :]

        # Concatenate sequence and mask
        combined = torch.cat([tcr_emb, epitope_emb], dim=1)
        key_padding_mask = torch.cat([tcr_mask, epitope_mask], dim=1)

        for layer in self.transformer_layers:
            combined = layer(combined, key_padding_mask=key_padding_mask)

        # Combine mean and max pooling
        pooled_mean = combined.mean(dim=1)
        pooled_max, _ = combined.max(dim=1)
        pooled = torch.cat([pooled_mean, pooled_max, trbv_embed, trbj_embed, mhc_embed], dim=1)

        output = self.classifier(pooled).squeeze(1)
        return output