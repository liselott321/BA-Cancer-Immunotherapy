import torch
import torch.nn as nn

# --- Residual Block ---
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
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

# --- Classifier ---
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
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

# --- Transformer Attention Block ---
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

# --- Dataset ---
class LazyTCR_Epitope_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_frame, tcr_embeddings, epitope_embeddings):
        self.data_frame = data_frame
        self.tcr_embeddings = tcr_embeddings
        self.epitope_embeddings = epitope_embeddings

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sample = self.data_frame.iloc[idx]
        tcr_id = sample['TRB_CDR3']
        epitope_id = sample['Epitope']
        label = sample['Binding']

        tcr_embedding = self.tcr_embeddings[tcr_id][:]
        epitope_embedding = self.epitope_embeddings[epitope_id][:]

        return (
            torch.tensor(tcr_embedding, dtype=torch.float32),
            torch.tensor(epitope_embedding, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )

# --- Full Model ---
class HybridTCR_Epitope_Model(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=2,
                 max_tcr_len=20, max_epi_len=15,
                 dropout=0.1, classifier_hidden_dim=64,
                 physico_feature_dim=0):  # ← Für spätere Erweiterung
        super().__init__()

        self.embed_dim = embed_dim
        self.seq_len = max_tcr_len + max_epi_len

        # Embedding layers for TCR & Epitope
        self.tcr_embed = nn.Linear(1024, embed_dim)
        self.epitope_embed = nn.Linear(1024, embed_dim)

        # Positional Encoding
        self.tcr_pos = nn.Parameter(torch.randn(1, max_tcr_len, embed_dim))
        self.epi_pos = nn.Parameter(torch.randn(1, max_epi_len, embed_dim))

        # Transformer encoder
        self.encoder = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])

        # Classifier input size = mean+max pooling + optional extra features
        self.classifier_input_dim = embed_dim * 2 + physico_feature_dim
        self.classifier = Classifier(self.classifier_input_dim, classifier_hidden_dim, dropout)

    def forward(self, tcr, epitope, physico_feats=None):
        # Linear embedding + Positional Encoding
        tcr = self.tcr_embed(tcr) + self.tcr_pos[:, :tcr.size(1), :]
        epi = self.epitope_embed(epitope) + self.epi_pos[:, :epitope.size(1), :]

        # Padding mask (optional)
        tcr_mask = (tcr.sum(dim=-1) == 0)
        epi_mask = (epitope.sum(dim=-1) == 0)
        key_padding_mask = torch.cat([tcr_mask, epi_mask], dim=1)

        # Concatenate sequence
        x = torch.cat([tcr, epi], dim=1)

        # Pass through Transformer
        for layer in self.encoder:
            x = layer(x, key_padding_mask=key_padding_mask)

        # Pooling
        pooled_mean = x.mean(dim=1)
        pooled_max, _ = x.max(dim=1)
        combined = torch.cat([pooled_mean, pooled_max], dim=1)

        # Optionally add physico-chemical feature vectors
        if physico_feats is not None:
            combined = torch.cat([combined, physico_feats], dim=1)

        return self.classifier(combined).squeeze(1)
