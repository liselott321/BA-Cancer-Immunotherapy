import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_output, _ = self.attn(
            x.permute(1, 0, 2),  # Query
            x.permute(1, 0, 2),  # Key
            x.permute(1, 0, 2),  # Value
            key_padding_mask=key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_output.permute(1, 0, 2)))
        return x

class TCR_Epitope_Dataset(Dataset):
    def __init__(self, data, tcr_embeddings, epitope_embeddings):
        self.data = data
        self.tcr_embeddings = tcr_embeddings
        self.epitope_embeddings = epitope_embeddings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tcr_embedding = torch.tensor(self.tcr_embeddings[row['TRB_CDR3']], dtype=torch.float32)
        epitope_embedding = torch.tensor(self.epitope_embeddings[row['Epitope']], dtype=torch.float32)
        label = torch.tensor(row['Binding'], dtype=torch.float32)  
        return tcr_embedding, epitope_embedding, label


<<<<<<< HEAD
# Custom Dataset class to handle lazy-loaded embeddings
=======
# for global descriptors
class LazyTCR_Epitope_Descriptor_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, tcr_emb, epi_emb, descriptor_file):
        self.df = df
        self.tcr_emb = tcr_emb
        self.epi_emb = epi_emb
        self.desc_data = h5py.File(descriptor_file, 'r')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tcr_embed = self.tcr_emb[row["TRB_CDR3"]][:]
        epi_embed = self.epi_emb[row["Epitope"]][:]
        tcr_desc = self.desc_data["tcr_encoded"][idx]
        epi_desc = self.desc_data["epi_encoded"][idx]
        label = row["Binding"]

        return (
            torch.tensor(tcr_embed, dtype=torch.float32),
            torch.tensor(epi_embed, dtype=torch.float32),
            torch.tensor(tcr_desc, dtype=torch.float32),
            torch.tensor(epi_desc, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )

class TCR_Epitope_Transformer_WithDescriptors(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, tcr_descriptor_dim, epi_descriptor_dim, dropout=0.1):
        super().__init__()

        self.protbert_tcr = nn.Sequential(
        nn.Linear(1024, embed_dim),
        nn.Tanh())
        
        self.protbert_epi = nn.Sequential(
            nn.Linear(1024, embed_dim),
            nn.Tanh()
        )
        self.desc_tcr = nn.Sequential(
            nn.Linear(tcr_descriptor_dim, embed_dim),
            nn.Tanh()
        )
        self.desc_epi = nn.Sequential(
            nn.Linear(epi_descriptor_dim, embed_dim),
            nn.Tanh())
    
        self.transformer_layers = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(embed_dim, 1)

    def forward(self, tcr_embed, epi_embed, tcr_desc, epi_desc):
        # Berechne ProtBERT Embeddings [B, 43, D]
        tcr_seq = self.protbert_tcr(tcr_embed)  # [B, 43, D]
        epi_seq = self.protbert_epi(epi_embed)  # [B, 43, D]
        
        # Deskriptoren sind global (also nur [B, D])
        # → expandiere auf Sequenzlänge
        tcr_desc_exp = self.desc_tcr(tcr_desc).unsqueeze(1).repeat(1, tcr_seq.shape[1], 1)  # [B, 43, D]
        epi_desc_exp = self.desc_epi(epi_desc).unsqueeze(1).repeat(1, epi_seq.shape[1], 1)  # [B, 43, D]
        
        # Kombiniere
        tcr = tcr_seq + tcr_desc_exp
        epi = epi_seq + epi_desc_exp

        combined = torch.cat((tcr, epi), dim=1)  # [B, 86, D]
        mask = (combined.sum(dim=-1) == 0)

        for layer in self.transformer_layers:
            combined = layer(combined, key_padding_mask=mask)

        pooled = combined.mean(dim=1)
        return self.output_layer(pooled).squeeze(1)



'''
>>>>>>> 94654f12c31d780adcda8273b594d94ecb205f52
class LazyTCR_Epitope_PLE_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, tcr_emb, epi_emb, ple_file):
        self.df = df
        self.tcr_emb = tcr_emb
        self.epi_emb = epi_emb
        self.ple_data = h5py.File(ple_file, 'r')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tcr_id = row["TRB_CDR3"]
        epi_id = row["Epitope"]
        label = row["Binding"]

        tcr_embed = self.tcr_emb[tcr_id][:]
        epi_embed = self.epi_emb[epi_id][:]
        tcr_ple = self.ple_data["tcr_ple"][idx]
        epi_ple = self.ple_data["epi_ple"][idx]

        return (
            torch.tensor(tcr_embed, dtype=torch.float32),
            torch.tensor(epi_embed, dtype=torch.float32),
            torch.tensor(tcr_ple, dtype=torch.float32),
            torch.tensor(epi_ple, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )

class TCR_Epitope_Transformer_Multimodal(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, max_seq_len, ple_dim=30, dropout=0.1):
        super().__init__()
        
        self.protbert_tcr = nn.Linear(512, embed_dim)
        self.protbert_epi = nn.Linear(512, embed_dim)
        self.ple_tcr = nn.Linear(ple_dim, embed_dim)
        self.ple_epi = nn.Linear(ple_dim, embed_dim)

        self.tcr_pos = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        self.epi_pos = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))

        self.transformer_layers = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(embed_dim, 1)

    def forward(self, tcr_embed, epi_embed, tcr_ple, epi_ple):
        # Physico durch Linear schicken (reshape auf [B*L, 30] → dann wieder zurück auf [B, L, embed_dim])
        tcr_ple_proj = self.ple_tcr(tcr_ple.view(-1, tcr_ple.shape[-1])).view(tcr_ple.shape[0], tcr_ple.shape[1], -1)
        epi_ple_proj = self.ple_epi(epi_ple.view(-1, epi_ple.shape[-1])).view(epi_ple.shape[0], epi_ple.shape[1], -1)
        
        # ProtBERT + Physico kombinieren
        tcr = self.protbert_tcr(tcr_embed) + tcr_ple_proj
        epi = self.protbert_epi(epi_embed) + epi_ple_proj

        # Add positional encodings
        tcr += self.tcr_pos[:, :tcr.size(1), :]
        epi += self.epi_pos[:, :epi.size(1), :]

        # Combine + masking
        combined = torch.cat((tcr, epi), dim=1)
        mask = (combined.sum(dim=-1) == 0)

        for layer in self.transformer_layers:
            combined = layer(combined, key_padding_mask=mask)

        pooled = combined.mean(dim=1)
<<<<<<< HEAD
        return self.output_layer(pooled).squeeze(1)
=======
        return self.output_layer(pooled).squeeze(1)'''
>>>>>>> 94654f12c31d780adcda8273b594d94ecb205f52
