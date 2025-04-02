import torch
import torch.nn as nn

class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(SelfAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, key_padding_mask=None):
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm.permute(1, 0, 2), x_norm.permute(1, 0, 2), x_norm.permute(1, 0, 2), key_padding_mask=key_padding_mask)
        x = x + attn_output.permute(1, 0, 2)
        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, q, k, v, key_padding_mask=None):
        q_norm = self.norm(q)
        k_norm = self.norm(k)
        v_norm = self.norm(v)
        attn_output, _ = self.attn(q_norm.permute(1, 0, 2), k_norm.permute(1, 0, 2), v_norm.permute(1, 0, 2), key_padding_mask=key_padding_mask)
        return attn_output.permute(1, 0, 2)

# Custom Dataset class to handle lazy-loaded embeddings
class LazyTCR_Epitope_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_frame, tcr_embeddings, epitope_embeddings):
        """
        Args:
            data_frame (DataFrame): The DataFrame containing sample data.
            tcr_embeddings (h5py.File): HDF5 file containing TCR embeddings.
            epitope_embeddings (h5py.File): HDF5 file containing epitope embeddings.
        """
        self.data_frame = data_frame
        self.tcr_embeddings = tcr_embeddings
        self.epitope_embeddings = epitope_embeddings

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sample = self.data_frame.iloc[idx]
        tcr_id = sample['TRB_CDR3']  # Column name for TCR IDs
        epitope_id = sample['Epitope']  # Column name for epitope IDs
        label = sample['Binding']  # Target label

        # Access embeddings lazily
        tcr_embedding = self.tcr_embeddings[tcr_id][:]
        epitope_embedding = self.epitope_embeddings[epitope_id][:]

        return (
            torch.tensor(tcr_embedding, dtype=torch.float32),
            torch.tensor(epitope_embedding, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )

class TCR_Epitope_Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, max_tcr_length, max_epitope_length, dropout=0.1):
        super(TCR_Epitope_Transformer, self).__init__()
        self.tcr_embedding = nn.Linear(1024, embed_dim)
        self.epitope_embedding = nn.Linear(1024, embed_dim)
        self.tcr_positional_encoding = nn.Parameter(torch.randn(1, max_tcr_length, embed_dim))
        self.epitope_positional_encoding = nn.Parameter(torch.randn(1, max_epitope_length, embed_dim))
        
        self.tcr_self_attention = nn.ModuleList([SelfAttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)])
        self.epitope_self_attention = nn.ModuleList([SelfAttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)])
        self.cross_attention = CrossAttentionBlock(embed_dim, num_heads, dropout)
        
        self.output_layer = nn.Linear(embed_dim, 1)

    def forward(self, tcr, epitope, tcr_mask=None, epitope_mask=None):

        # Compute embeddings
        tcr_embedded = self.tcr_embedding(tcr)
        epitope_embedded = self.epitope_embedding(epitope)

        # Create key padding mask BEFORE adding positional encodings
        tcr_mask = (tcr.sum(dim=-1) == 0)  # Mask positions where TCR values are zero
        epitope_mask = (epitope.sum(dim=-1) == 0)  # Mask positions where epitope values are zero

        # Add positional encodings
        tcr = tcr_embedded + self.tcr_positional_encoding[:, :tcr_embedded.size(1), :]
        epitope = epitope_embedded + self.epitope_positional_encoding[:, :epitope_embedded.size(1), :]


        # tcr = self.tcr_embedding(tcr) + self.tcr_positional_encoding[:, :tcr.size(1), :]
        # epitope = self.epitope_embedding(epitope) + self.epitope_positional_encoding[:, :epitope.size(1), :]
        
        # for layer in self.tcr_self_attention:
        #     tcr = layer(tcr, key_padding_mask=tcr_mask)
        
        # for layer in self.epitope_self_attention:
        #     epitope = layer(epitope, key_padding_mask=epitope_mask)
        
        
        # Concatenate TCR and epitope along the sequence length dimension
        combined = torch.cat((tcr, epitope), dim=1)
        key_padding_mask = torch.cat((tcr_mask, epitope_mask), dim=1)  # Combine TCR and epitope masks

        for layer in self.tcr_self_attention:
            combined = layer(combined, key_padding_mask=key_padding_mask)

        tcr = self.cross_attention(tcr, epitope, epitope, key_padding_mask=epitope_mask)
        epitope = self.cross_attention(epitope, tcr, tcr, key_padding_mask=tcr_mask)


        combined_2 = torch.cat((tcr, epitope), dim=1)
        combined_3 = torch.cat((combined, combined_2), dim=1).mean(dim=1)
        output = self.output_layer(combined_3).squeeze(1)
        return output
