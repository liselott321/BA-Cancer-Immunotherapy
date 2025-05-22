import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score, average_precision_score, roc_curve, precision_recall_curve, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import pandas as pd
import sys
import yaml
import h5py
import wandb
from dotenv import load_dotenv
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# Import the enhanced model
from models.morning_stars_v1.beta.v6_1024_all_features_enhanced import TCR_Epitope_Transformer_Enhanced, LazyFullFeatureDataset, BidirectionalCrossAttention, FeatureStratifiedSampler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.arg_parser import * # pars_args

args = parse_args()
####  &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

def analyze_dataset_features(dataframe, name="Dataset"):
    """Analyze dataset and extract key features for stratification."""
    print(f"\n=== {name} Analysis ===")
    
    # Overall binding distribution
    binding_counts = dataframe["Binding"].value_counts()
    binding_percentage = 100 * binding_counts / len(dataframe)
    print(f"Binding distribution:")
    for label, count in binding_counts.items():
        print(f"  {'Binding' if label == 1 else 'Non-binding'}: {count} ({binding_percentage[label]:.1f}%)")
    
    # Add feature columns if they don't exist
    if "TCR_length" not in dataframe.columns:
        dataframe["TCR_length"] = dataframe["TRB_CDR3"].str.len()
    if "Epitope_length" not in dataframe.columns:
        dataframe["Epitope_length"] = dataframe["Epitope"].str.len()
    
    # TCR/Epitope length distribution
    print(f"\nTCR length: min={dataframe['TCR_length'].min()}, max={dataframe['TCR_length'].max()}, mean={dataframe['TCR_length'].mean():.1f}")
    print(f"Epitope length: min={dataframe['Epitope_length'].min()}, max={dataframe['Epitope_length'].max()}, mean={dataframe['Epitope_length'].mean():.1f}")
    
    # MHC class distribution
    mhc_counts = dataframe["MHC"].value_counts().head(5)
    print("\nMHC distribution (top 5):")
    for mhc, count in mhc_counts.items():
        mhc_binding = dataframe[dataframe["MHC"] == mhc]["Binding"].mean() * 100
        print(f"  {mhc}: {count} examples ({mhc_binding:.1f}% binding)")
    
    # Create MHC Class feature (if not already present)
    if "MHC_Class" not in dataframe.columns:
        # Directly apply the logic without using a nested function
        # Using pandas' built-in methods to avoid needing pd in nested scope
        
        # First, create a series of "Unknown" values
        mhc_class = pd.Series(["Unknown"] * len(dataframe), index=dataframe.index)
        
        # Then use boolean indexing to update values based on conditions
        # For non-null values:
        non_null_mask = dataframe["MHC"].notna()
        
        # For MHC-I class
        mhc1_mask = dataframe["MHC"].str.startswith(("HLA-A", "HLA-B", "HLA-C"), na=False)
        mhc_class[mhc1_mask] = "MHC-I"
        
        # For MHC-II class
        mhc2_mask = dataframe["MHC"].str.startswith("HLA-D", na=False)
        mhc_class[mhc2_mask] = "MHC-II"
        
        # For other non-null values that don't match the above
        other_mask = non_null_mask & ~mhc1_mask & ~mhc2_mask
        mhc_class[other_mask] = "Other"
        
        # Assign back to dataframe
        dataframe["MHC_Class"] = mhc_class
    
    # Report MHC class distribution
    mhc_class_counts = dataframe["MHC_Class"].value_counts()
    print("\nMHC Class distribution:")
    for mhc_class, count in mhc_class_counts.items():
        mhc_class_binding = dataframe[dataframe["MHC_Class"] == mhc_class]["Binding"].mean() * 100
        print(f"  {mhc_class}: {count} examples ({mhc_class_binding:.1f}% binding)")
    
    # Classify TCR sequences into rough categories 
    if "TCR_Category" not in dataframe.columns:
        # Using similar approach as for MHC_Class to avoid nested function issues
        
        # Default category is "Unknown"
        tcr_category = pd.Series(["Unknown"] * len(dataframe), index=dataframe.index)
        
        # For non-null, non-empty values:
        valid_tcr_mask = dataframe["TRB_CDR3"].notna() & (dataframe["TRB_CDR3"].str.len() > 0)
        
        # Apply complex logic using pandas methods
        # 1. Extract the first 3 characters (or all if less than 3)
        motifs = dataframe.loc[valid_tcr_mask, "TRB_CDR3"].apply(
            lambda x: x[:3] if len(x) >= 3 else x
        )
        
        # 2. Determine length category
        lengths = dataframe.loc[valid_tcr_mask, "TRB_CDR3"].apply(
            lambda x: "Short" if len(x) < 12 else "Medium" if len(x) < 16 else "Long"
        )
        
        # 3. Combine to make the category
        tcr_category[valid_tcr_mask] = motifs + "_" + lengths
        
        # Assign back to dataframe
        dataframe["TCR_Category"] = tcr_category
    
    # Get top TCR categories
    tcr_cat_counts = dataframe["TCR_Category"].value_counts().head(5)
    print("\nTCR Categories (top 5):")
    for cat, count in tcr_cat_counts.items():
        cat_binding = dataframe[dataframe["TCR_Category"] == cat]["Binding"].mean() * 100
        print(f"  {cat}: {count} examples ({cat_binding:.1f}% binding)")
    
    return dataframe
####  &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
## **********************************************************************************************************************
def generate_enhanced_negatives_v2(dataframe, n_samples=1000):
    """Generate negative samples using sequence properties, ensuring we can get embeddings."""
    positives = dataframe[dataframe["Binding"] == 1].copy()
    
    if len(positives) == 0:
        print("No positive samples found to create negatives from")
        return pd.DataFrame()
    
    all_negatives = []
    
    # Strategy 1: Pair TCRs with epitopes from different MHC contexts
    # This strategy works well because we're using existing TCRs and epitopes
    print("Generating negatives by MHC context shuffling...")
    mhc_groups = positives.groupby("MHC")
    if len(mhc_groups) > 1:  # Only if we have multiple MHC types
        n_mhc_samples = n_samples // 2  # Increase proportion for this strategy
        
        for mhc, group in mhc_groups:
            if len(group) < 2:  # Skip if too few examples
                continue
                
            # For each example in this MHC group
            for _, row in group.sample(min(len(group), n_mhc_samples // len(mhc_groups)), replace=False).iterrows():
                # Find epitopes from different MHC contexts
                other_mhcs = positives[positives["MHC"] != mhc]
                if len(other_mhcs) == 0:
                    continue
                    
                # Select a random epitope from a different MHC
                other_epitope_row = other_mhcs.sample(1).iloc[0]
                
                new_row = row.copy()
                new_row["Epitope"] = other_epitope_row["Epitope"]
                new_row["Binding"] = 0
                new_row["Negative_Type"] = "MHC_Shuffled"
                all_negatives.append(new_row)
    
    # Strategy 2: TCR substitution - pair epitopes with TCRs that bind to very different epitopes
    # This also works well as we're using existing TCRs and epitopes
    print("Generating negatives by TCR substitution...")
    n_tcr_samples = n_samples // 2  # Increase proportion for this strategy
    
    # Group by epitope length - a rough proxy for similarity
    epitope_length_groups = positives.groupby(positives["Epitope"].str.len())
    
    for length, length_group in epitope_length_groups:
        if len(length_group) < 5:  # Skip if too few examples
            continue
            
        for _, row in length_group.sample(min(len(length_group), n_tcr_samples // len(epitope_length_groups)), replace=False).iterrows():
            # Find TCRs that bind to epitopes of very different lengths
            different_length = positives[abs(positives["Epitope"].str.len() - length) > 3]
            if len(different_length) == 0:
                continue
                
            # Select a random TCR from a different epitope length group
            other_row = different_length.sample(1).iloc[0]
            
            new_row = row.copy()
            new_row["TRB_CDR3"] = other_row["TRB_CDR3"]
            new_row["TRBV"] = other_row["TRBV"]
            new_row["TRBJ"] = other_row["TRBJ"]
            new_row["Binding"] = 0
            new_row["Negative_Type"] = "TCR_Substitution"
            all_negatives.append(new_row)
    
    # Convert to dataframe
    if not all_negatives:
        print("Warning: No negative samples could be generated")
        return pd.DataFrame()
    
    negatives_df = pd.DataFrame(all_negatives)
    
    # Add a feature column to mark these as generated negatives
    negatives_df["Generated_Negative"] = 1
    
    # Ensure we don't duplicate existing pairs
    existing_pairs = set(zip(dataframe["TRB_CDR3"], dataframe["Epitope"]))
    negatives_df = negatives_df[~negatives_df.apply(lambda x: (x["TRB_CDR3"], x["Epitope"]) in existing_pairs, axis=1)]
    
    # Limit to requested number
    if len(negatives_df) > n_samples:
        negatives_df = negatives_df.sample(n_samples, replace=False)
    
    print(f"Generated {len(negatives_df)} new negative samples")
    return negatives_df

## *************************************************************************************************************************

# Load Configurations
with open(args.configs_path, "r") as file:
    config = yaml.safe_load(file)

epochs = args.epochs if args.epochs else config['epochs']
batch_size = args.batch_size if args.batch_size else config['batch_size']
print(f'Batch size: {batch_size}')
learning_rate = args.learning_rate if args.learning_rate else config['learning_rate']
print(f'Learning rate: {learning_rate}')

train_path = args.train if args.train else config['data_paths']['train']
print(f"train_path: {train_path}")
val_path = args.val if args.val else config['data_paths']['val']
print(f"val_path: {val_path}")

# physchem_path = config['embeddings']['physchem']
physchem_path= "../../../../data/physico/descriptor_encoded_physchem.h5"
physchem_file = h5py.File(physchem_path, 'r')

# path to save best model
model_path = args.model_path if args.model_path else config['model_path']
# Directory for model checkpoints
checkpoint_dir = os.path.join(os.path.dirname(model_path), "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# Logging setup
PROJECT_NAME = "dataset-allele"
ENTITY_NAME = "ba_cancerimmunotherapy"
MODEL_NAME = "v6_enhanced_fs_fbbs_ens"
experiment_name = f"Experiment - {MODEL_NAME}"
run_name = f"Run_{os.path.basename(model_path).replace('.pth', '')}"
run = wandb.init(project=PROJECT_NAME, job_type=f"{experiment_name}", entity="ba_cancerimmunotherapy", name=run_name, config=config)

# Log hyperparameters explicitly
wandb.config.update({
    "model_name": MODEL_NAME,
    "embed_dim": config["embed_dim"],
    "max_tcr_length": config["max_tcr_length"],
    "max_epitope_length": config["max_epitope_length"],
    "use_checkpointing": True,  # Add new parameters for enhanced model
    "bidirectional_cross_attention": True
})

# Embeddings paths from config/args
tcr_train_path = args.tcr_train_embeddings if args.tcr_train_embeddings else config['embeddings']['tcr_train']
epitope_train_path = args.epitope_train_embeddings if args.epitope_train_embeddings else config['embeddings']['epitope_train']
tcr_valid_path = args.tcr_valid_embeddings if args.tcr_valid_embeddings else config['embeddings']['tcr_valid']
epitope_valid_path = args.epitope_valid_embeddings if args.epitope_valid_embeddings else config['embeddings']['epitope_valid']

# Load Data -------------------------------------------------------
dataset_name = f"beta_allele"
artifact = wandb.use_artifact("ba_cancerimmunotherapy/dataset-allele/beta_allele:latest")
data_dir = artifact.download(f"./WnB_Experiments_Datasets/{dataset_name}")
    
train_file_path = f"{data_dir}/allele/train.tsv"
val_file_path = f"{data_dir}/allele/validation.tsv"

train_data = pd.read_csv(train_file_path, sep="\t")
val_data = pd.read_csv(val_file_path, sep="\t")

physchem_map = pd.read_csv("../../../../data/physico/descriptor_encoded_physchem_mapping.tsv", sep="\t")

# Per Sequenz joinen
train_data = pd.merge(train_data, physchem_map, on=["TRB_CDR3", "Epitope"], how="left")
val_data = pd.merge(val_data, physchem_map, on=["TRB_CDR3", "Epitope"], how="left")

# Mappings erstellen
trbv_dict = {v: i for i, v in enumerate(train_data["TRBV"].unique())}
trbj_dict = {v: i for i, v in enumerate(train_data["TRBJ"].unique())}
mhc_dict  = {v: i for i, v in enumerate(train_data["MHC"].unique())}

UNKNOWN_TRBV_IDX = len(trbv_dict)
UNKNOWN_TRBJ_IDX = len(trbj_dict)
UNKNOWN_MHC_IDX  = len(mhc_dict)

for df in [train_data, val_data]:
    df["TRBV_Index"] = df["TRBV"].map(trbv_dict).fillna(UNKNOWN_TRBV_IDX).astype(int)
    df["TRBJ_Index"] = df["TRBJ"].map(trbj_dict).fillna(UNKNOWN_TRBJ_IDX).astype(int)
    df["MHC_Index"]  = df["MHC"].map(mhc_dict).fillna(UNKNOWN_MHC_IDX).astype(int)

# Vokabulargrößen bestimmen
trbv_vocab_size = UNKNOWN_TRBV_IDX + 1
trbj_vocab_size = UNKNOWN_TRBJ_IDX + 1
mhc_vocab_size  = UNKNOWN_MHC_IDX + 1

print(trbv_vocab_size)
print(trbj_vocab_size)
print(mhc_vocab_size)

# Load Embeddings -------------------------------------------------------
# HDF5 Lazy Loading for embeddings
def load_h5_lazy(file_path):
    """Lazy load HDF5 file and return a reference to the file."""
    return h5py.File(file_path, 'r')

print('Loading embeddings...')
print("tcr_train ", tcr_train_path)
tcr_train_embeddings = load_h5_lazy(tcr_train_path)
print("epi_train ", epitope_train_path)
epitope_train_embeddings = load_h5_lazy(epitope_train_path)
print("tcr_valid ", tcr_valid_path)
tcr_valid_embeddings = load_h5_lazy(tcr_valid_path)
print("epi_valid ", epitope_valid_path)
epitope_valid_embeddings = load_h5_lazy(epitope_valid_path)

emb_physchem_path = "../../../../data/physico/descriptor_encoded_physchem.h5"  # change if not server 1

with h5py.File(emb_physchem_path, 'r') as f:
    inferred_physchem_dim = f["tcr_encoded"].shape[1]

##### -------------------------- NEW STRATIFICATION STRATEGY
# 1. First analyze your training and validation datasets
train_data = analyze_dataset_features(train_data, "Training")
val_data = analyze_dataset_features(val_data, "Validation")

# 2. Generate enhanced negatives for training
print("\nGenerating enhanced negative samples...")
n_enhanced_negatives = min(5000, len(train_data[train_data["Binding"] == 1]))
enhanced_negatives = generate_enhanced_negatives_v2(
    train_data, 
    n_samples=n_enhanced_negatives
)
# After generating enhanced negatives
# 3. Add enhanced negatives to training data
# After generating enhanced negatives
if not enhanced_negatives.empty:
    print(f"Adding {len(enhanced_negatives)} enhanced negatives to training data")
    
    # First add the enhanced negatives to the training data
    train_data = pd.concat([train_data, enhanced_negatives], ignore_index=True)
    
    # Update the physicochemical mappings for ALL training data
    print(f"Updating physicochemical mappings for combined dataset of {len(train_data)} samples")
    physchem_map = pd.read_csv("../../../../data/physico/descriptor_encoded_physchem_mapping.tsv", sep="\t")
    
    # This will map physicochemical properties for all samples, including our new ones if they exist
    train_data = pd.merge(train_data, physchem_map, on=["TRB_CDR3", "Epitope"], how="left")
    
    # Check for missing mappings in the entire dataset  physchem_index
    missing_physchem = train_data[train_data["physchem_index"].isna()]
    if len(missing_physchem) > 0:
        print(f"Warning: {len(missing_physchem)} samples missing physicochemical mappings")
        print(f"Of these, {missing_physchem.get('Generated_Negative', 0).sum()} are from our generated negatives")
        
        # Drop samples with missing physchem data
        train_data = train_data.dropna(subset=["physchem_index"])
        print(f"Keeping {len(train_data)} samples with valid physicochemical mappings")



# 4. Create your datasets
train_dataset = LazyFullFeatureDataset(train_data, tcr_train_embeddings, epitope_train_embeddings, physchem_file, trbv_dict, trbj_dict, mhc_dict)
val_dataset = LazyFullFeatureDataset(val_data, tcr_valid_embeddings, epitope_valid_embeddings, physchem_file, trbv_dict, trbj_dict, mhc_dict)

# 5. Create feature-stratified sampler
stratified_sampler = FeatureStratifiedSampler(train_dataset, train_data, batch_size=batch_size)


class RotatingFullCoverageSampler:
    def __init__(self, dataset, labels, batch_size=32):
        self.dataset = dataset
        self.labels = np.array(labels)
        self.batch_size = batch_size

        self.pos_indices = np.where(self.labels == 1)[0]
        self.neg_indices = np.where(self.labels == 0)[0]

        self.pos_pointer = 0
        self.neg_pointer = 0

        np.random.shuffle(self.pos_indices)
        np.random.shuffle(self.neg_indices)

    def get_loader(self):
        chunk_size = min(len(self.pos_indices) - self.pos_pointer, len(self.neg_indices) - self.neg_pointer)

        if chunk_size == 0:
            # Reset when everything has been used at least once
            self.pos_pointer = 0
            self.neg_pointer = 0
            np.random.shuffle(self.pos_indices)
            np.random.shuffle(self.neg_indices)
            chunk_size = min(len(self.pos_indices), len(self.neg_indices))

        sampled_pos = self.pos_indices[self.pos_pointer:self.pos_pointer + chunk_size]
        sampled_neg = self.neg_indices[self.neg_pointer:self.neg_pointer + chunk_size]

        self.pos_pointer += chunk_size
        self.neg_pointer += chunk_size

        combined = np.concatenate([sampled_pos, sampled_neg])
        np.random.shuffle(combined)

        subset = Subset(self.dataset, combined)
        return DataLoader(subset, batch_size=self.batch_size, shuffle=True)

num_pos = len(train_data[train_data["Binding"] == 1])
num_neg = len(train_data[train_data["Binding"] == 0])
max_pairs_per_epoch = min(num_pos, num_neg) # Da immer nur gleich viele Positives und Negatives ziehen (1:1)
required_epochs = math.ceil(max(num_pos, num_neg) / max_pairs_per_epoch)
print(f"Mindestens {required_epochs} Epochen nötig, um alle Daten einmal zu verwenden.")

# Data loaders
train_labels = train_data['Binding'].values 
balanced_generator = RotatingFullCoverageSampler(train_dataset, train_labels, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

dropout = args.dropout if args.dropout else config['dropout']

model = TCR_Epitope_Transformer_Enhanced(
    embed_dim=config['embed_dim'],
    num_heads=config['num_heads'],
    num_layers=config['num_layers'],
    max_tcr_length=config['max_tcr_length'],
    max_epitope_length=config['max_epitope_length'],
    dropout=dropout,
    physchem_dim=inferred_physchem_dim,
    trbv_vocab_size=trbv_vocab_size,
    trbj_vocab_size=trbj_vocab_size,
    mhc_vocab_size=mhc_vocab_size,
    use_checkpointing=True  # Set to False if memory isn't an issue 
).to(device)

wandb.watch(model, log="all", log_freq=100)

# Loss
pos_count = (train_labels == 1).sum()
neg_count = (train_labels == 0).sum()
pos_weight = torch.tensor([neg_count / pos_count]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Automatically load sweep configuration into local variables
learning_rate = args.learning_rate if args.learning_rate else wandb.config.learning_rate
batch_size = args.batch_size if args.batch_size else wandb.config.batch_size
optimizer_name = args.optimizer or wandb.config.get("optimizer", config.get("optimizer", "adam"))
num_layers = args.num_layers if args.num_layers else wandb.config.num_layers
num_heads = args.num_heads if args.num_heads else wandb.config.num_heads
weight_decay = args.weight_decay or wandb.config.get("weight_decay", config.get("weight_decay", 0.0))

if optimizer_name == "adam":
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
elif optimizer_name == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
else:
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)
    
best_ap = 0.0
best_model_state = None
early_stop_counter = 0
min_epochs = required_epochs 
patience = 4
global_step = 0

# Function to save model checkpoint to wandb - FIXED VERSION
def save_checkpoint(model, epoch, optimizer, scheduler=None):
    """Save model checkpoint with only serializable components"""
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
    
    # Only save the essential components that are serializable
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    # Add scheduler state if it exists
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    # Log checkpoint to wandb
    artifact = wandb.Artifact(f"model_checkpoint_epoch_{epoch+1}", type="model")
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact)
    
    return checkpoint_path

# Training Loop ---------------------------------------------------------------
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    
    # Number of batches to use per epoch - adjusted for stratified sampler
    batches_per_epoch = max(50, len(train_data) // batch_size)
    
    # Use tqdm for progress tracking
    train_loader_tqdm = tqdm(range(batches_per_epoch), desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False)
    
    for batch_idx in train_loader_tqdm:
        # Get a balanced batch using the stratified sampler
        train_loader = stratified_sampler.get_loader()
        try:
            # Get a single batch from the loader
            train_batch = next(iter(train_loader))
        except StopIteration:
            print("Warning: StopIteration in stratified sampler. This should not happen.")
            continue
            
        # Move batch to device
        tcr, epitope, tcr_phys, epi_phys, trbv, trbj, mhc, label = (
            tcr.to(device),
            epitope.to(device),
            tcr_phys.to(device),
            epi_phys.to(device),
            trbv.to(device), trbj.to(device), mhc.to(device),
            label.to(device),
        )
        
        # Forward and backward passes
        optimizer.zero_grad()
        output = model(tcr, epitope, tcr_phys, epi_phys, trbv, trbj, mhc)
        loss = criterion(output, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        epoch_loss += loss.item()
        batch_loss = epoch_loss / (batch_idx + 1)
        
        # Log metrics
        train_loader_tqdm.set_postfix(loss=batch_loss)
        wandb.log({"train_loss": loss.item(), "epoch": epoch}, step=global_step)
        global_step += 1


        train_loader_tqdm.set_postfix(loss=epoch_loss / (train_loader_tqdm.n + 1))

    # Validation --------------------------------------------------------------------------------------------------------------
    model.eval()
    all_labels = []
    all_outputs = []
    all_preds = []

    val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]", leave=False)
    val_loss_total = 0

    with torch.no_grad():
        for tcr, epitope, tcr_phys, epi_phys, trbv, trbj, mhc, label in val_loader_tqdm:
            tcr, epitope, tcr_phys, epi_phys, trbv, trbj, mhc, label = (
                tcr.to(device),
                epitope.to(device),
                tcr_phys.to(device),
                epi_phys.to(device),
                trbv.to(device), trbj.to(device), mhc.to(device),
                label.to(device),
            )
            output = model(tcr, epitope, tcr_phys, epi_phys, trbv, trbj, mhc)
            val_loss = criterion(output, label)
            val_loss_total += val_loss.item()

            # Convert logits to probabilities and predictions
            probs = torch.sigmoid(output)
            preds = (probs > 0.5).float()

            all_labels.extend(label.cpu().numpy())
            all_outputs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    precision_curve, recall_curve, thresholds = precision_recall_curve(all_labels, all_outputs)
    # F1 Score berechnen für alle Thresholds
    f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = np.max(f1_scores)
    
    print(f"Best threshold (by F1): {best_threshold:.4f} with F1: {best_f1:.4f}")
    wandb.log({"best_threshold": best_threshold, "best_f1_score_from_curve": best_f1}, step=global_step, commit=False)
    
    # Jetzt F1, Accuracy, Precision, Recall etc. mit best_threshold berechnen
    preds = (all_outputs > best_threshold).astype(float)
    
    # Convert to NumPy arrays for metric calculations
    all_labels = np.array(all_labels)
    all_preds = np.array(preds)
    all_outputs = np.array(all_outputs)

    # Metrics
    auc = roc_auc_score(all_labels, all_outputs)
    ap = average_precision_score(all_labels, all_outputs)
    accuracy = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds)
    scheduler.step(auc)
    wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]}, step=global_step, commit=False)

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss/len(train_loader):.4f}, Val Loss: {val_loss_total/len(val_loader):.4f}, Val AUC: {auc:.4f}, Val AP: {ap:.4f}, Val Accuracy: {accuracy:.4f}, Val F1: {f1:.4f}, TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_outputs)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    

    # ---------------------new metrics per MHC-class

    # After calculating the global metrics in validation:
    # Feature-based validation analysis
    print("\n=== Feature-Based Validation Analysis ===")

    # 1. Analysis by MHC Class
    if "MHC_Class" in val_data.columns:
        print("\nPerformance by MHC Class:")
        mhc_classes = val_data["MHC_Class"].unique()
        for mhc_class in mhc_classes:
            mask = val_data["MHC_Class"] == mhc_class
            if mask.sum() < 10:  # Skip with too few examples
                continue
                
            class_indices = np.where(mask)[0]
            if len(class_indices) >= 10:
                # Get predictions for this class
                class_labels = all_labels[mask]
                class_outputs = all_outputs[mask]
                class_preds = all_preds[mask]
                
                if len(np.unique(class_labels)) < 2:
                    print(f"  {mhc_class}: Skipping metrics (only one class present)")
                    continue
                    
                # Calculate metrics
                class_auc = roc_auc_score(class_labels, class_outputs)
                class_ap = average_precision_score(class_labels, class_outputs)
                class_f1 = f1_score(class_labels, class_preds)
                class_accuracy = accuracy_score(class_labels, class_preds)
                
                print(f"  {mhc_class} ({len(class_indices)} examples, {class_labels.mean()*100:.1f}% binding)")
                print(f"    AUC: {class_auc:.4f}, AP: {class_ap:.4f}")
                print(f"    F1: {class_f1:.4f}, Accuracy: {class_accuracy:.4f}")
                
                # Log to wandb
                wandb.log({
                    f"val_{mhc_class}_auc": class_auc,
                    f"val_{mhc_class}_ap": class_ap,
                    f"val_{mhc_class}_f1": class_f1,
                    f"val_{mhc_class}_accuracy": class_accuracy
                }, step=global_step, commit=False)

    # 2. Analysis by TCR Length
    if "TCR_length" in val_data.columns:
        print("\nPerformance by TCR Length:")
        # Define TCR length bins
        tcr_bins = [0, 10, 13, 16, 100]
        tcr_bin_names = ["≤10", "11-13", "14-16", "≥17"]
        
        for i in range(len(tcr_bins)-1):
            min_len, max_len = tcr_bins[i], tcr_bins[i+1]
            bin_name = tcr_bin_names[i]
            
            mask = (val_data["TCR_length"] > min_len) & (val_data["TCR_length"] <= max_len)
            if mask.sum() < 10:
                continue
                
            bin_labels = all_labels[mask]
            bin_outputs = all_outputs[mask]
            bin_preds = all_preds[mask]
            
            if len(np.unique(bin_labels)) < 2:
                continue
                
            bin_auc = roc_auc_score(bin_labels, bin_outputs)
            bin_ap = average_precision_score(bin_labels, bin_outputs)
            bin_f1 = f1_score(bin_labels, bin_preds)
            bin_accuracy = accuracy_score(bin_labels, bin_preds)
            
            print(f"  Length {bin_name} ({mask.sum()} examples, {bin_labels.mean()*100:.1f}% binding)")
            print(f"    AUC: {bin_auc:.4f}, AP: {bin_ap:.4f}")
            print(f"    F1: {bin_f1:.4f}, Accuracy: {bin_accuracy:.4f}")
            
            # Log to wandb
            wandb.log({
                f"val_tcr_length_{bin_name}_auc": bin_auc,
                f"val_tcr_length_{bin_name}_ap": bin_ap,
                f"val_tcr_length_{bin_name}_f1": bin_f1,
                f"val_tcr_length_{bin_name}_accuracy": bin_accuracy
            }, step=global_step, commit=False)

    # ----- until here

    # Speichern und in wandb loggen
    os.makedirs("results", exist_ok=True)
    roc_curve_path = f"results/roc_curve_epoch_{epoch+1}.png"
    plt.savefig(roc_curve_path)
    wandb.log({"roc_curve": wandb.Image(roc_curve_path)}, step=global_step, commit=False)
    plt.close()

    wandb.log({
        "epoch": epoch + 1,
        "train_loss_epoch": epoch_loss / len(train_loader),
        "val_loss": val_loss_total / len(val_loader),
        "val_auc": auc,
        "val_ap": ap,
        "val_f1": f1,
        "val_accuracy": accuracy,
        "val_tp": tp,
        "val_tn": tn,
        "val_fp": fp,
        "val_fn": fn,
        "val_precision": precision,
        "val_recall": recall,
        "prediction_distribution": wandb.Histogram(all_outputs),
        "label_distribution": wandb.Histogram(all_labels),
        "val_confusion_matrix": wandb.plot.confusion_matrix(
            y_true=all_labels,
            preds=all_preds,
            class_names=["Not Binding", "Binding"])
    }, step=global_step, commit=False)

    # ===== TPP1–TPP4 Auswertung im Validierungsset =====
    if "task" in val_data.columns:
        all_tasks = val_data["task"].values

        for tpp in ["TPP1", "TPP2", "TPP3", "TPP4"]:
            mask = all_tasks == tpp
            if mask.sum() > 0:
                labels = all_labels[mask]
                outputs = all_outputs[mask]
                preds = all_preds[mask]

                unique_classes = np.unique(labels)

                if len(unique_classes) == 2:
                    tpp_auc = roc_auc_score(labels, outputs)
                    tpp_ap = average_precision_score(labels, outputs)
                else:
                    tpp_auc = None
                    tpp_ap = None
                    print(f"  {tpp}: Nur eine Klasse vorhanden – AUC & AP übersprungen.")

                tpp_f1 = f1_score(labels, preds, zero_division=0)
                tpp_acc = accuracy_score(labels, preds)
                tpp_precision = precision_score(labels, preds, zero_division=0)
                tpp_recall = recall_score(labels, preds, zero_division=0)

                print(f"\n    {tpp} ({mask.sum()} Beispiele)")
                print(f"AUC:  {tpp_auc if tpp_auc is not None else 'n/a'}")
                print(f"AP:   {tpp_ap if tpp_ap is not None else 'n/a'}")
                print(f"F1:   {tpp_f1:.4f}")
                print(f"Acc:  {tpp_acc:.4f}")
                print(f"Precision: {tpp_precision:.4f}")
                print(f"Recall:    {tpp_recall:.4f}")

                log_dict = {
                    f"val_{tpp}_f1": tpp_f1,
                    f"val_{tpp}_accuracy": tpp_acc,
                    f"val_{tpp}_precision": tpp_precision,
                    f"val_{tpp}_recall": tpp_recall,
                }
                if tpp_auc is not None:
                    log_dict[f"val_{tpp}_auc"] = tpp_auc
                if tpp_ap is not None:
                    log_dict[f"val_{tpp}_ap"] = tpp_ap

                wandb.log(log_dict, step=global_step, commit=False)

                wandb.log({
                    f"val_{tpp}_confusion_matrix": wandb.plot.confusion_matrix(
                        y_true=labels.astype(int),
                        preds=preds.astype(int),
                        class_names=["Not Binding", "Binding"],
                        title=f"Confusion Matrix – {tpp}"
                    )
                }, step=global_step, commit=False)
                # Histogramm der Modellkonfidenz (Vorhersagewahrscheinlichkeiten)
                plt.figure(figsize=(6, 4))
                plt.hist(outputs, bins=50, color='skyblue', edgecolor='black')
                plt.title(f"Prediction Score Distribution – {tpp}")
                plt.xlabel("Predicted Probability")
                plt.ylabel("Frequency")
                plt.tight_layout()
                
                # Speicherpfad & Logging
                plot_path = f"results/{tpp}_confidence_hist_epoch{epoch+1}.png"
                plt.savefig(plot_path)
                wandb.log({f"val_{tpp}_prediction_distribution": wandb.Image(plot_path)}, step=global_step, commit=False)
                plt.close()
            else:
                print(f"\n Keine Beispiele für {tpp} im Validationset.")
    else:
        print("\n Keine Spalte 'task' in val_data – TPP-Auswertung übersprungen.")
    
    # Save model checkpoint after each epoch (FIXED VERSION)
    checkpoint_path = save_checkpoint(model, epoch, optimizer, scheduler)
    print(f"Saved checkpoint for epoch {epoch+1} to {checkpoint_path}")

    # Early Stopping: nur auf multiples von `min_epochs` schauen
    if ap > best_ap:
        best_ap = ap
        best_model_state = model.state_dict()
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"No improvement in AP. Early stop counter: {early_stop_counter}/{patience}")
    
    # Check: nur abbrechen, wenn epoch ein Vielfaches von min_epochs ist UND patience erreicht ist
    if ((epoch + 1) % min_epochs == 0) and early_stop_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch+1}.")
        break

# Save best model -------------------------------------------------------------------------------
if best_model_state:
    os.makedirs("results/trained_models/v6_all_features_pe", exist_ok=True)
    torch.save(best_model_state, model_path)
    print("Best model saved with AP:", best_ap)

    artifact = wandb.Artifact(run_name + "_best_model", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

wandb.finish()
print("Best Hyperparameters:")
print(wandb.config)