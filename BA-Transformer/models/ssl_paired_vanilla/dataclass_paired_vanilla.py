import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class PairedVanilla(Dataset):
    def __init__(self, dataset_path, embed_base_path, traV, traJ, trbV, trbJ, mhc, transform=None):
        """
        Dataclass for the experiment NOT USING (!) physicochemical properties.
        """
        
        self.dataset_path = dataset_path
        self.epitope_embeddings_path = f"{embed_base_path}/Epitope_paired_embeddings.npz"
        self.tra_embeddings_path = f"{embed_base_path}/TRA_paired_embeddings.npz"
        self.trb_embeddings_path = f"{embed_base_path}/TRB_paired_embeddings.npz"
        self.pretrain = pretrain

        # Neue physikochemische Merkmale laden
        precision = 'allele'
        base_path = os.path.expanduser("~/BA/BA_ZHAW/data/physicoProperties")
        physico_path = f"{base_path}/ssl_paired/{precision}"
        self.epitope_physico_path = f"{physico_path}/Epitope_physico.npz"
        self.tra_physico_path = f"{physico_path}/TRA_CDR3_physico.npz"
        self.trb_physico_path = f"{physico_path}/TRB_CDR3_physico.npz"
        
        self.transform = transform
        self.data_frame = pd.read_csv(self.dataset_path, sep='\t')

        # Lade physikochemische Eigenschaften
        epitope_physico = np.load(self.epitope_physico_path)
        tra_physico = np.load(self.tra_physico_path)
        trb_physico = np.load(self.trb_physico_path)

        # Normalisierung der physikochemischen Eigenschaften
        self.normalize_physicochemical_features(epitope_physico, tra_physico, trb_physico)

        # Lade embeddings
        epitope_embeddings = np.load(self.epitope_embeddings_path)
        tra_embeddings = np.load(self.tra_embeddings_path)
        trb_embeddings = np.load(self.trb_embeddings_path)

        traV_dict = traV
        traJ_dict = traJ
        trbV_dict = trbV
        trbJ_dict = trbJ
        mhc_dict = mhc
        
        self.data_frame["Epitope Embedding"] = self.data_frame["Epitope"].map(epitope_embeddings)
        self.data_frame["TRA_CDR3 Embedding"] = self.data_frame["TRA_CDR3"].map(tra_embeddings)
        self.data_frame["TRB_CDR3 Embedding"] = self.data_frame["TRB_CDR3"].map(trb_embeddings)

        # Füge die physikochemischen Merkmale hinzu
        self.data_frame["Epitope Physicochemical"] = self.data_frame["Epitope"].map(epitope_physico)
        self.data_frame["TRA_CDR3 Physicochemical"] = self.data_frame["TRA_CDR3"].map(tra_physico)
        self.data_frame["TRB_CDR3 Physicochemical"] = self.data_frame["TRB_CDR3"].map(trb_physico)

        self.data_frame["TRAV Index"] = self.data_frame["TRAV"].map(traV_dict)
        self.data_frame["TRAJ Index"] = self.data_frame["TRAJ"].map(traJ_dict)
        self.data_frame["TRBV Index"] = self.data_frame["TRBV"].map(trbV_dict)
        self.data_frame["TRBJ Index"] = self.data_frame["TRBJ"].map(trbJ_dict)
        self.data_frame["MHC Index"] = self.data_frame["MHC"].map(mhc_dict)

        columns = list(self.data_frame.columns) 
        columns.remove('Binding')
        columns.append('Binding')
        self.data_frame = self.data_frame[columns]

        # Check for unmapped TRA_CDR3 sequences
        unmapped_tra = self.data_frame[self.data_frame["TRA_CDR3 Embedding"].isna()]["TRA_CDR3"].unique()
        print("Unmapped TRA_CDR3 sequences:", len(unmapped_tra), unmapped_tra[:10])
        
        # Check for unmapped TRB_CDR3 sequences
        unmapped_trb = self.data_frame[self.data_frame["TRB_CDR3 Embedding"].isna()]["TRB_CDR3"].unique()
        print("Unmapped TRB_CDR3 sequences:", len(unmapped_trb), unmapped_trb[:10])
        
        # Check for unmapped Epitope sequences
        unmapped_epitope = self.data_frame[self.data_frame["Epitope Embedding"].isna()]["Epitope"].unique()
        print("Unmapped Epitope sequences:", len(unmapped_epitope), unmapped_epitope[:10])

    def normalize_physicochemical_features(self, epitope_physico, tra_physico, trb_physico):
        # Epitope Physicochemical
        epitope_physico_data = {key: epitope_physico[key] for key in epitope_physico.files}
        tra_physico_data = {key: tra_physico[key] for key in tra_physico.files}
        trb_physico_data = {key: trb_physico[key] for key in trb_physico.files}
    
        # Normalisiere die Epitope Daten
        for key in epitope_physico_data:
            feature_array = epitope_physico_data[key]
            feature_min = feature_array.min()
            feature_max = feature_array.max()
            epitope_physico_data[key] = (feature_array - feature_min) / (feature_max - feature_min)
    
        # Normalisiere die TRA Physicochemical Daten
        for key in tra_physico_data:
            feature_array = tra_physico_data[key]
            feature_min = feature_array.min()
            feature_max = feature_array.max()
            tra_physico_data[key] = (feature_array - feature_min) / (feature_max - feature_min)
    
        # Normalisiere die TRB Physicochemical Daten
        for key in trb_physico_data:
            feature_array = trb_physico_data[key]
            feature_min = feature_array.min()
            feature_max = feature_array.max()
            trb_physico_data[key] = (feature_array - feature_min) / (feature_max - feature_min)
    
        # Update the self.data_frame with normalized features
        self.data_frame["Epitope Physicochemical"] = self.data_frame["Epitope"].map(epitope_physico_data)
        self.data_frame["TRA_CDR3 Physicochemical"] = self.data_frame["TRA_CDR3"].map(tra_physico_data)
        self.data_frame["TRB_CDR3 Physicochemical"] = self.data_frame["TRB_CDR3"].map(trb_physico_data)


    def __len__(self):
        return len(self.data_frame)

    
    def __getitem__(self, index):
        row = self.data_frame.iloc[index]  

        # Lade die physikochemischen Merkmale
        epitope_physicochemical = torch.tensor(row["Epitope Physicochemical"], dtype=torch.float32)
        tra_physicochemical = torch.tensor(row["TRA_CDR3 Physicochemical"], dtype=torch.float32)
        trb_physicochemical = torch.tensor(row["TRB_CDR3 Physicochemical"], dtype=torch.float32)
    
        # Kombiniere physikochemische Merkmale zu einem Gesamtvektor
        physicochemical_features = torch.cat([epitope_physicochemical, tra_physicochemical, trb_physicochemical])
    
        # Wenn im Vortrainingsmodus, gib nur die physikochemischen Merkmale zurück
        if self.pretrain:
            return {
                "physicochemical_features": physicochemical_features
            }

        epitope_embedding = torch.tensor(row["Epitope Embedding"], dtype=torch.float32)
        epitope_sequence = row["Epitope"]
        tra_cdr3_embedding = torch.tensor(row["TRA_CDR3 Embedding"], dtype=torch.float32)
        tra_cdr3_sequence = row["TRA_CDR3"]
        trb_cdr3_embedding = torch.tensor(row["TRB_CDR3 Embedding"], dtype=torch.float32)
        trb_cdr3_sequence = row["TRB_CDR3"]
        
        # Weitere Batch-Daten
        v_alpha = row["TRAV Index"]
        j_alpha = row["TRAJ Index"]
        v_beta = row["TRBV Index"]
        j_beta = row["TRBJ Index"]
        mhc = row["MHC Index"]
        task = row["task"]
        
        label = torch.tensor(row["Binding"], dtype=torch.float)

        # Erstelle den Batch, inklusive physikochemischer Merkmale
        batch = {
            "epitope_embedding": epitope_embedding,
            "epitope_sequence": epitope_sequence,
            "tra_cdr3_embedding": tra_cdr3_embedding,
            "tra_cdr3_sequence": tra_cdr3_sequence,
            "trb_cdr3_embedding": trb_cdr3_embedding,
            "trb_cdr3_sequence": trb_cdr3_sequence,
            "physicochemical_features": physicochemical_features,
            "v_alpha": v_alpha,
            "j_alpha": j_alpha,
            "v_beta": v_beta,
            "j_beta": j_beta,
            "mhc": mhc,
            "task": task,
            "label": label,
        }

        if self.transform:
            batch = self.transform(batch)

        return batch
    

    def get_max_length(self):
        df = self.data_frame
        df = df[["Epitope Embedding", "TRA_CDR3 Embedding", "TRB_CDR3 Embedding"]]
        epitope_embeddings = []
        tra_cdr3_embeddings = []
        trb_cdr3_embeddings = []

        for i, row in df.iterrows():
            epitope_embeddings.append(row["Epitope Embedding"])
            tra_cdr3_embeddings.append(row["TRA_CDR3 Embedding"])
            trb_cdr3_embeddings.append(row["TRB_CDR3 Embedding"])

        max_length = max(
            max(embedding.shape[0] for embedding in epitope_embeddings),
            max(embedding.shape[0] for embedding in tra_cdr3_embeddings),
            max(embedding.shape[0] for embedding in trb_cdr3_embeddings)
        )

        return max_length
        