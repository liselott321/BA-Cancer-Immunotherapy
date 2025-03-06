import os
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# Gerät (GPU/CPU) automatisch erkennen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CPU oder GPU:", device)


# Multi-Head Attention Mechanismus für Joint Embedding - Simple V1
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        return attn_output


# Laden der Sequenz-Embeddings
def load_embeddings(embedding_file):
    """Lädt gespeicherte Embeddings als NumPy-Array"""
    data = np.load(embedding_file, allow_pickle=True)
    return data


# Padding-Funktion für ungleich lange Sequenzen (GPU-kompatibel)
def pad_embeddings(embedding_list, max_length, padding_value=0.0):
    """Padding für eine Liste von Embeddings, um gleiche Länge zu erhalten"""
    padded_list = []
    
    for emb in embedding_list:
        emb_tensor = torch.tensor(emb, dtype=torch.float32, device=device)  # Direkt auf GPU
        pad_size = max_length - emb_tensor.shape[0]  # Anzahl der fehlenden Zeilen
        
        if pad_size > 0:
            # Füge am Ende Padding mit 0 hinzu
            padding = torch.full((pad_size, emb_tensor.shape[1]), padding_value, dtype=torch.float32, device=device)
            emb_tensor = torch.cat((emb_tensor, padding), dim=0)
        
        padded_list.append(emb_tensor)
    
    return torch.stack(padded_list)  # Tensor bleibt auf GPU


# Dataset-Klasse für PyTorch DataLoader (GPU-kompatibel)
class TCREpitopeDataset(Dataset):
    def __init__(self, tcr_embeddings, epitope_embeddings):
        self.tcr_embeddings = list(tcr_embeddings.values())
        self.epitope_embeddings = list(epitope_embeddings.values())

        # Bestimme die maximale Sequenzlänge für Padding
        self.max_length = max(
            max(len(emb) for emb in self.tcr_embeddings),
            max(len(emb) for emb in self.epitope_embeddings)
        )

    def __len__(self):
        return len(self.tcr_embeddings)

    def __getitem__(self, idx):
        tcr_embedding = self.tcr_embeddings[idx]
        epitope_embedding = self.epitope_embeddings[idx]

        # Padding direkt auf GPU durchführen
        tcr_padded = pad_embeddings([tcr_embedding], self.max_length)
        epitope_padded = pad_embeddings([epitope_embedding], self.max_length)

        # Joint Embedding erstellen (TCR + Epitope)
        joint_embedding = torch.cat((tcr_padded.squeeze(0), epitope_padded.squeeze(0)), dim=0).to(device)

        return joint_embedding


# Richtige Pfade für die Embeddings setzen
base_path = os.path.expanduser("~/data/embeddings/beta/gene/")
tcr_embedding_path = os.path.join(base_path, "TRB_beta_embeddings.npz")
epitope_embedding_path = os.path.join(base_path, "Epitope_beta_embeddings.npz")

# Laden der TCR- & Epitope-Embeddings
tcr_embeddings = load_embeddings(tcr_embedding_path)
epitope_embeddings = load_embeddings(epitope_embedding_path)

print("Anzahl TCR-Embeddings:", len(tcr_embeddings))
print("Anzahl Epitope-Embeddings:", len(epitope_embeddings))

# Dataset & DataLoader für Batch-Verarbeitung
batch_size = 64
dataset = TCREpitopeDataset(tcr_embeddings, epitope_embeddings)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)  # 🚀 Wichtiger Fix für GPU-Zugriff

# Initialisiere Multi-Head Attention
embedding_dim = 1024  # ProtBERT-Embeddings haben oft 1024 Dimensionen
num_heads = 8  # muss teilbar durch 1024 sein... 
multihead_attn = MultiHeadSelfAttention(embed_dim=embedding_dim, num_heads=num_heads).to(device)
print("Läuft MultiHead Attention auf der GPU?", next(multihead_attn.parameters()).is_cuda)


# Batchweise Verarbeitung
for batch_idx, joint_batch in enumerate(dataloader):
    print(f"Batch {batch_idx+1} Verarbeitung...")

    # Überprüfe, wo sich das Batch befindet (CPU oder GPU)
    print("Vor .to(device): Joint Batch ist auf:", joint_batch.device)  # Sollte 'cpu' sein

    # Sicherstellen, dass Daten auf GPU sind
    joint_batch = joint_batch.to(device, non_blocking=True)

    # Nochmals prüfen, ob das Batch auf der GPU ist
    print("Nach .to(device): Joint Batch ist auf:", joint_batch.device)  # Sollte 'cuda:0' sein

    # Multi-Head Attention anwenden
    attn_output = multihead_attn(joint_batch)

    # PRÜFEN, OB ALLES KORREKT IST (nur für den ersten Batch zur Kontrolle)
    if batch_idx == 0:
        print("Joint Input Shape:", joint_batch.shape)  # Erwartet: (batch, seq_len, embed_dim)
        print("Attention Output Shape:", attn_output.shape)  # Sollte (batch, seq_len, embed_dim) sein

        # Mittelwert & Standardabweichung prüfen
        print("Attention Mean:", attn_output.mean().cpu().item())
        print("Attention Std:", attn_output.std().cpu().item())

    # Speicher optimieren
    del joint_batch, attn_output
    torch.cuda.empty_cache()



'''
#PHYSICO HINZUFüGEN - ACHTUNG NOCH NICHT VERWENDET, REINE CODE IDEE
from ple_embedding import ple_encode_sequence  # Importiere den PLE-Encoder
# Beispielhafte Epitope & TCR-Sequenzen
tcr_sequence = "CASSLGTGANYGYTF"
epitope_sequence = "GILGFVFTL"
# PLE-Embedding berechnen
tcr_physico = torch.tensor(ple_encode_sequence(tcr_sequence), dtype=torch.float32)
epitope_physico = torch.tensor(ple_encode_sequence(epitope_sequence), dtype=torch.float32)
# Kombiniere Attention-Ausgabe mit physikochemischen Features
final_features = torch.cat((fused_features, tcr_physico, epitope_physico), dim=1)
'''