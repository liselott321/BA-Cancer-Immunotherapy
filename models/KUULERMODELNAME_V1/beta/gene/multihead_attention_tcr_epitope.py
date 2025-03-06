import os
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel


# Gerät (GPU/CPU) automatisch erkennen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CPU oder GPU:", device)


# Multi-Head Attention Mechanismus: Simple
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
	@@ -25,14 +23,26 @@ def forward(self, x):
        return attn_output


# Laden der Sequenz-Embeddings (ProtBERT)
def load_embeddings(embedding_file):
    """ Lädt gespeicherte Embeddings als NumPy-Array """
    data = np.load(embedding_file, allow_pickle=True)
    return data


# Dataset-Klasse für PyTorch DataLoader
class TCREpitopeDataset(Dataset):
    def __init__(self, tcr_embeddings, epitope_embeddings):
        self.tcr_embeddings = list(tcr_embeddings.values())
        self.epitope_embeddings = list(epitope_embeddings.values())

    def __len__(self):
        return len(self.tcr_embeddings)

    def __getitem__(self, idx):
        return torch.tensor(self.tcr_embeddings[idx], dtype=torch.float32), \
               torch.tensor(self.epitope_embeddings[idx], dtype=torch.float32)


# Padding auf die maximale Sequenzlänge setzen
def pad_embeddings(embedding_list, padding_value=0.0):
	@@ -42,64 +52,63 @@ def pad_embeddings(embedding_list, padding_value=0.0):
    return padded_embeddings.to(device)


# Richtige Pfade für die Embeddings setzen
base_path = os.path.expanduser("~/data/embeddings/beta/gene/")  # GENE und BETA
tcr_embedding_path = os.path.join(base_path, "TRB_beta_embeddings.npz")
epitope_embedding_path = os.path.join(base_path, "Epitope_beta_embeddings.npz")


# Laden der TCR- & Epitope-Embeddings
tcr_embeddings = load_embeddings(tcr_embedding_path)
epitope_embeddings = load_embeddings(epitope_embedding_path)

print("Anzahl TCR-Embeddings:", len(tcr_embeddings))
print("Anzahl Epitope-Embeddings:", len(epitope_embeddings))


# Dataset & DataLoader für Batch-Verarbeitung
dataset = TCREpitopeDataset(tcr_embeddings, epitope_embeddings)
batch_size = 64  # Anpassen je nach GPU-Speicher
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Initialisiere Multi-Head Attention
embedding_dim = 1024  # ProtBERT-Embeddings haben oft 1024 Dimensionen
num_heads = 8  # Anzahl der Attention-Heads
multihead_attn = MultiHeadSelfAttention(embed_dim=embedding_dim, num_heads=num_heads).to(device)


# Batchweise Verarbeitung
for batch_idx, (tcr_batch, epitope_batch) in enumerate(dataloader):
    print(f"Batch {batch_idx+1} Verarbeitung...")

    # Daten auf GPU verschieben
    tcr_batch = tcr_batch.to(device)
    epitope_batch = epitope_batch.to(device)

    # Multi-Head Attention anwenden
    tcr_attn_output = multihead_attn(tcr_batch)
    epitope_attn_output = multihead_attn(epitope_batch)

    # Features fusionieren
    fused_features = torch.cat((tcr_attn_output, epitope_attn_output), dim=1).to(device)

    # PRÜFEN, OB ALLES KORREKT IST (nur für den ersten Batch zur Kontrolle)
    if batch_idx == 0:
        print("TCR Input Shape:", tcr_batch.shape)  # Erwartet: (batch, seq_len, embed_dim)
        print("Epitope Input Shape:", epitope_batch.shape)

        print("TCR Attention Output Shape:", tcr_attn_output.shape)  # Sollte (batch, seq_len, embed_dim) sein
        print("Epitope Attention Output Shape:", epitope_attn_output.shape)

        # Mittelwert & Standardabweichung prüfen
        print("TCR Attention Mean:", tcr_attn_output.mean().cpu().item())
        print("TCR Attention Std:", tcr_attn_output.std().cpu().item())

    # Speicher optimieren
    del tcr_batch, epitope_batch, tcr_attn_output, epitope_attn_output
    torch.cuda.empty_cache()