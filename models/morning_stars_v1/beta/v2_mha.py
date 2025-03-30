import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_output, _ = self.attn(
            x.permute(1, 0, 2),
            x.permute(1, 0, 2),
            x.permute(1, 0, 2),
            key_padding_mask=key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_output.permute(1, 0, 2)))
        return x

class TCR_Epitope_Dataset(Dataset):
    def __init__(self, data, tcr_embeddings, epitope_embeddings):
        self.data = data
        self.tcr_embeddings = tcr_embeddings
        self.epitope_embeddings = epitope_embeddings

        self.trbv_dict = {val: idx for idx, val in enumerate(data['TRBV'].unique())}
        self.trbj_dict = {val: idx for idx, val in enumerate(data['TRBJ'].unique())}
        self.mhc_dict = {val: idx for idx, val in enumerate(data['MHC'].unique())}

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tcr_embedding = torch.tensor(self.tcr_embeddings[row['TRB_CDR3']], dtype=torch.float32)
        epitope_embedding = torch.tensor(self.epitope_embeddings[row['Epitope']], dtype=torch.float32)
        label = torch.tensor(row['Binding'], dtype=torch.float32)

        trbv = self.trbv_dict[row['TRBV']]
        trbj = self.trbj_dict[row['TRBJ']]
        mhc = self.mhc_dict[row['MHC']]
        task = row['task']
    
        return tcr_embedding, epitope_embedding, label, trbv, trbj, mhc, task

class LazyTCR_Epitope_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_frame, tcr_embeddings, epitope_embeddings, trbv_dict, trbj_dict, mhc_dict):
        self.data_frame = data_frame
        self.tcr_embeddings = tcr_embeddings
        self.epitope_embeddings = epitope_embeddings
        self.trbv_dict = trbv_dict
        self.trbj_dict = trbj_dict
        self.mhc_dict = mhc_dict

    def __len__(self):
        return len(self.data_frame)
        
    def __getitem__(self, idx):
        sample = self.data_frame.iloc[idx]
        tcr_id = sample['TRB_CDR3']
        epitope_id = sample['Epitope']
        label = sample['Binding']
        task = sample['task']
    
        tcr_embedding = self.tcr_embeddings[tcr_id][:]
        epitope_embedding = self.epitope_embeddings[epitope_id][:]
    
        trbv = self.trbv_dict.get(sample['TRBV'], 0)
        trbj = self.trbj_dict.get(sample['TRBJ'], 0)
        mhc = self.mhc_dict.get(sample['MHC'], 0)
<<<<<<< HEAD
=======

        if sample["TRBV"] not in self.trbv_dict:
            print(f"Fehlender TRBV: {sample['TRBV']}")
>>>>>>> 94654f12c31d780adcda8273b594d94ecb205f52
    
        return (
            torch.tensor(tcr_embedding, dtype=torch.float32),
            torch.tensor(epitope_embedding, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(trbv, dtype=torch.long),
            torch.tensor(trbj, dtype=torch.long),
            torch.tensor(mhc, dtype=torch.long),
            task
        )
        
class TCR_Epitope_Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, max_tcr_length, max_epitope_length,
                 num_trbv, num_trbj, num_mhc, dropout=0.1):
        super(TCR_Epitope_Transformer, self).__init__()

        self.tcr_embedding = nn.Linear(512, embed_dim)
        self.epitope_embedding = nn.Linear(512, embed_dim)

        self.trbv_embedding = nn.Embedding(num_trbv, embed_dim)
        self.trbj_embedding = nn.Embedding(num_trbj, embed_dim)
        self.mhc_embedding = nn.Embedding(num_mhc, embed_dim)

        self.tcr_positional_encoding = nn.Parameter(torch.randn(1, max_tcr_length, embed_dim))
        self.epitope_positional_encoding = nn.Parameter(torch.randn(1, max_epitope_length, embed_dim))

        self.transformer_layers = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(embed_dim, 1)

    def forward(self, tcr, epitope, trbv, trbj, mhc):
        # Embedding
        tcr_embedded = self.tcr_embedding(tcr)
        epitope_embedded = self.epitope_embedding(epitope)

        # Positional encoding
        tcr = tcr_embedded + self.tcr_positional_encoding[:, :tcr_embedded.size(1), :]
        epitope = epitope_embedded + self.epitope_positional_encoding[:, :epitope_embedded.size(1), :]

        # Lookup: TRBV, TRBJ, MHC (shape: [batch_size, embed_dim])
        trbv_embed = self.trbv_embedding(trbv).unsqueeze(1)
        trbj_embed = self.trbj_embedding(trbj).unsqueeze(1)
        mhc_embed = self.mhc_embedding(mhc).unsqueeze(1)

        # Combine all into one sequence: [batch_size, total_seq_len, embed_dim]
        combined = torch.cat((tcr, epitope, trbv_embed, trbj_embed, mhc_embed), dim=1)

        # Key padding mask (optional – here just set to None)
        key_padding_mask = None

        # Transformer encoding
        for layer in self.transformer_layers:
            combined = layer(combined, key_padding_mask=key_padding_mask)

        # Mean pooling & output
        pooled = combined.mean(dim=1)
        output = self.output_layer(pooled).squeeze(1)
        return output