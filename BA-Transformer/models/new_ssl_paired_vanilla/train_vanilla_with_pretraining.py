import os
import wandb
import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, StochasticWeightAveraging
from vanilla_model import VanillaModel
from dataclass_paired_vanilla import PairedVanilla
from sklearn.preprocessing import StandardScaler

# Hyperparameters
MODEL_NAME = "VanillaModelWithPretraining"
EMBEDDING_SIZE = 1024
BATCH_SIZE = 64
EPOCHS = 250
NUM_WORKERS = 0
PRETRAIN_EPOCHS = 50

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def set_hyperparameters(config):
    hyperparameters = {}
    hyperparameters["optimizer"] = config.optimizer
    hyperparameters["learning_rate"] = config.learning_rate
    hyperparameters["weight_decay"] = config.weight_decay
    hyperparameters["dropout_attention"] = config.dropout_attention
    hyperparameters["dropout_linear"] = config.dropout_linear
    return hyperparameters

class PadCollate:
    def __init__(self, seq_max_length):
        self.seq_max_length = seq_max_length

    def pad_collate(self, batch):
        epitope_embeddings, tra_cdr3_embeddings, trb_cdr3_embeddings = [], [], []
        v_alpha, j_alpha, v_beta, j_beta = [], [], [], []
        epitope_sequence, tra_cdr3_sequence, trb_cdr3_sequence = [], [], []
        mhc = []
        task = []
        labels = []

        for item in batch:
            epitope_embeddings.append(item["epitope_embedding"])
            epitope_sequence.append(item["epitope_sequence"])
            tra_cdr3_embeddings.append(item["tra_cdr3_embedding"])
            tra_cdr3_sequence.append(item["tra_cdr3_sequence"])
            trb_cdr3_embeddings.append(item["trb_cdr3_embedding"])
            trb_cdr3_sequence.append(item["trb_cdr3_sequence"])
            v_alpha.append(item["v_alpha"])
            j_alpha.append(item["j_alpha"])
            v_beta.append(item["v_beta"])
            j_beta.append(item["j_beta"])
            mhc.append(item["mhc"])
            task.append(item["task"])
            labels.append(item["label"])

        max_length = self.seq_max_length

        def pad_embeddings(embeddings):
            return torch.stack([
                torch.nn.functional.pad(embedding, (0, 0, 0, max_length - embedding.size(0)), "constant", 0)
                for embedding in embeddings
            ])

        epitope_embeddings = pad_embeddings(epitope_embeddings)
        tra_cdr3_embeddings = pad_embeddings(tra_cdr3_embeddings)
        trb_cdr3_embeddings = pad_embeddings(trb_cdr3_embeddings)

        v_alpha = torch.tensor(v_alpha, dtype=torch.int32)
        j_alpha = torch.tensor(j_alpha, dtype=torch.int32)
        v_beta = torch.tensor(v_beta, dtype=torch.int32)
        j_beta = torch.tensor(j_beta, dtype=torch.int32)
        mhc = torch.tensor(mhc, dtype=torch.int32)

        labels = torch.stack(labels)

        return {
            "epitope_embedding": epitope_embeddings,
            "epitope_sequence": epitope_sequence,
            "tra_cdr3_embedding": tra_cdr3_embeddings,
            "tra_cdr3_sequence": tra_cdr3_sequence,
            "trb_cdr3_embedding": trb_cdr3_embeddings,
            "trb_cdr3_sequence": trb_cdr3_sequence,
            "v_alpha": v_alpha,
            "j_alpha": j_alpha,
            "v_beta": v_beta,
            "j_beta": j_beta,
            "mhc": mhc,
            "task": task,
            "label": labels
        }

def load_physicochemical_features(file_path):
    data = np.load(file_path)
    return data

def pretrain_with_physicochemical_features(model, dataloader, scaler, epitope_features, tra_features, trb_features):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    for epoch in range(PRETRAIN_EPOCHS):
        total_loss = 0
        for batch in dataloader:
            epitope_sequence = batch["epitope_sequence"]
            tra_cdr3_sequence = batch["tra_cdr3_sequence"]
            trb_cdr3_sequence = batch["trb_cdr3_sequence"]

            # Fetch features from loaded npz data
            epitope_features_batch = np.array([epitope_features[seq] for seq in epitope_sequence])
            tra_features_batch = np.array([tra_features[seq] for seq in tra_cdr3_sequence])
            trb_features_batch = np.array([trb_features[seq] for seq in trb_cdr3_sequence])

            # Convert to torch tensors
            features = torch.tensor(np.concatenate([epitope_features_batch, tra_features_batch, trb_features_batch]), dtype=torch.float32).to(DEVICE)

            # Determine batch size and feature dimension
            batch_size = features.size(0)
            feature_dim = features.size(1)

            # Dummy values for the additional arguments (ensure the correct shape)
            tra_cdr3 = torch.zeros((batch_size, feature_dim), dtype=torch.float32).to(DEVICE)
            trb_cdr3 = torch.zeros((batch_size, feature_dim), dtype=torch.float32).to(DEVICE)
            v_alpha = torch.zeros((batch_size,), dtype=torch.long).to(DEVICE)
            j_alpha = torch.zeros((batch_size,), dtype=torch.long).to(DEVICE)
            v_beta = torch.zeros((batch_size,), dtype=torch.long).to(DEVICE)
            j_beta = torch.zeros((batch_size,), dtype=torch.long).to(DEVICE)
            mhc = torch.zeros((batch_size,), dtype=torch.long).to(DEVICE)

            # Forward pass
            optimizer.zero_grad()
            output = model(features, tra_cdr3, trb_cdr3, v_alpha, j_alpha, v_beta, j_beta, mhc, is_pretraining=True)

            # Compute loss
            loss = criterion(output, features)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Speicher freigeben
            del epitope_sequence, tra_cdr3_sequence, trb_cdr3_sequence
            del epitope_features_batch, tra_features_batch, trb_features_batch
            del tra_cdr3, trb_cdr3, v_alpha, j_alpha, v_beta, j_beta, mhc, output, loss

            torch.cuda.empty_cache()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{PRETRAIN_EPOCHS}], Loss: {avg_loss:.4f}")
        wandb.log({"pretrain_loss": avg_loss})

def main():
    precision = "allele"
    embed_base_dir = f"../../data/embeddings/paired/{precision}"
    hyperparameter_tuning_with_WnB = False

    # W&B Setup
    experiment_name = f"Experiment - {MODEL_NAME}"
    PROJECT_NAME = f'dataset-{precision}'
    run = wandb.init(project=PROJECT_NAME, job_type=f"{experiment_name}", entity="pa_cancerimmunotherapy")
    config = wandb.config

    # Use the W&B artifact to download the dataset
    dataset_name = f"paired_{precision}"
    artifact = run.use_artifact(f"{dataset_name}:latest")
    data_dir = artifact.download(f"./WnB_Experiments_Datasets/paired_{precision}")

    # Update the paths to use the downloaded directory
    train_file_path = f"{data_dir}/allele/train.tsv"
    val_file_path = f"{data_dir}/allele/validation.tsv"
    test_file_path = f"{data_dir}/allele/test.tsv"

    # Load the data
    df_train = pd.read_csv(train_file_path, sep="\t")
    df_val = pd.read_csv(val_file_path, sep="\t")
    df_test = pd.read_csv(test_file_path, sep="\t")

    # Load physicochemical features from saved .npz files
    physico_base_path = "../../data/physicoProperties/ssl_paired/allele/"
    epitope_features = load_physicochemical_features(f"{physico_base_path}Epitope_physico.npz")
    tra_cdr3_features = load_physicochemical_features(f"{physico_base_path}TRA_CDR3_physico.npz")
    trb_cdr3_features = load_physicochemical_features(f"{physico_base_path}TRB_CDR3_physico.npz")

    # Pretraining Dataset
    traV_dict = {value: idx for idx, value in enumerate(df_train["TRAV"].unique())}
    traJ_dict = {value: idx for idx, value in enumerate(df_train["TRAJ"].unique())}
    trbV_dict = {value: idx for idx, value in enumerate(df_train["TRBV"].unique())}
    trbJ_dict = {value: idx for idx, value in enumerate(df_train["TRBJ"].unique())}
    mhc_dict = {value: idx for idx, value in enumerate(df_train["MHC"].unique())}

    train_dataset = PairedVanilla(train_file_path, embed_base_dir, traV_dict, traJ_dict, trbV_dict, trbJ_dict, mhc_dict)
    SEQ_MAX_LENGTH = max(train_dataset.get_max_length(), 1024)
    pad_collate = PadCollate(SEQ_MAX_LENGTH).pad_collate

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=RandomSampler(train_dataset),
        num_workers=NUM_WORKERS,
        collate_fn=pad_collate
    )

    # Model Setup
    hyperparameters = set_hyperparameters(config) if hyperparameter_tuning_with_WnB else {
        "optimizer": "sgd",
        "learning_rate": 5e-3,
        "weight_decay": 0.075,
        "dropout_attention": 0.3,
        "dropout_linear": 0.45
    }

    model = VanillaModel(EMBEDDING_SIZE, SEQ_MAX_LENGTH, DEVICE, len(traV_dict), len(traJ_dict), len(trbV_dict), len(trbJ_dict), len(mhc_dict), hyperparameters).to(DEVICE)

    # Pretraining Phase with physicochemical properties
    print("Starting pretraining phase with physicochemical features...")
    pretrain_with_physicochemical_features(model, train_dataloader, scaler=None, epitope_features=epitope_features, tra_features=tra_cdr3_features, trb_features=trb_cdr3_features)

    # Training Phase
    print("Starting main training phase...")
    wandb_logger = WandbLogger(project=PROJECT_NAME, name=experiment_name)
    wandb_logger.watch(model)
    tensorboard_logger = TensorBoardLogger("tb_logs", name=f"{MODEL_NAME}")

    model_checkpoint = ModelCheckpoint(dirpath="checkpoints", monitor="AP_Val", mode="max", save_top_k=1)
    early_stopping = EarlyStopping(monitor="AP_Val", patience=5, verbose=True, mode="max")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    swa = StochasticWeightAveraging(swa_lrs=hyperparameters["learning_rate"] * 0.1, swa_epoch_start=45)

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        logger=[wandb_logger, tensorboard_logger],
        callbacks=[model_checkpoint, early_stopping, lr_monitor, swa],
        accelerator="gpu"
    )
    trainer.fit(model, train_dataloader)

    # Save Model
    torch.save(model.state_dict(), f"{MODEL_NAME}.pth")
    print(f"Saved PyTorch Model State to {MODEL_NAME}.pth")

if __name__ == '__main__':
    main()
