import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
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
from models.morning_stars_v1.beta.v6_1024_all_features_pe import TCR_Epitope_Transformer_AllFeatures, LazyFullFeatureDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.arg_parser import parse_args

class EarlyStopping:
    """Early stopping to prevent overfitting based on validation F1-score"""
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score, model=None):
        if self.best_score is None:
            self.best_score = score
            if model and self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        
        improved = score > self.best_score + self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
            if model and self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
    
    def restore_best_weights_to_model(self, model):
        """Restore the best weights to the model"""
        if self.best_weights:
            model.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in self.best_weights.items()})

def create_balanced_sampler(labels, batch_size):
    """Create a WeightedRandomSampler for balanced training"""
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

def find_optimal_threshold(y_true, y_scores):
    """Find optimal classification threshold based on F1-score"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    if len(thresholds) > 0:
        optimal_idx = np.argmax(f1_scores)
        if optimal_idx < len(thresholds):
            return thresholds[optimal_idx], f1_scores[optimal_idx]
    
    return 0.5, 0.0

def evaluate_with_optimal_threshold(labels, outputs):
    """Evaluate predictions using optimal threshold"""
    optimal_thresh, best_f1_score = find_optimal_threshold(labels, outputs)
    
    # Make predictions with optimal threshold
    preds = (outputs > optimal_thresh).astype(float)
    
    # Calculate all metrics
    metrics = {
        'threshold': optimal_thresh,
        'f1': f1_score(labels, preds, zero_division=0),
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'auc': roc_auc_score(labels, outputs) if len(np.unique(labels)) == 2 else 0,
        'ap': average_precision_score(labels, outputs) if len(np.unique(labels)) == 2 else 0
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    metrics.update({'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp})
    
    return metrics, preds

def main():
    args = parse_args()

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

    physchem_path = config['embeddings']['physchem']
    physchem_file = h5py.File(physchem_path, 'r')

    # path to save best model
    model_path = args.model_path if args.model_path else config['model_path']

    # Logging setup
    PROJECT_NAME = "dataset-allele"
    ENTITY_NAME = "ba_cancerimmunotherapy"
    MODEL_NAME = "v6_all_features_pe_improved"
    experiment_name = f"Experiment - {MODEL_NAME}"
    run_name = f"Run_{os.path.basename(model_path).replace('.pt', '')}_balanced"
    run = wandb.init(project=PROJECT_NAME, job_type=f"{experiment_name}", entity="ba_cancerimmunotherapy", name=run_name, config=config)

    # Log hyperparameters explicitly
    wandb.config.update({
        "model_name": MODEL_NAME,
        "embed_dim": config["embed_dim"],
        "max_tcr_length": config["max_tcr_length"],
        "max_epitope_length": config["max_epitope_length"],
        "improved_training": True,
        "early_stopping": True,
        "optimal_threshold": True
    })

    # Embeddings paths from config/args
    tcr_train_path = args.tcr_train_embeddings if args.tcr_train_embeddings else config['embeddings']['tcr_train']
    epitope_train_path = args.epitope_train_embeddings if args.epitope_train_embeddings else config['embeddings']['epitope_train']
    tcr_valid_path = args.tcr_valid_embeddings if args.tcr_valid_embeddings else config['embeddings']['tcr_valid']
    epitope_valid_path = args.epitope_valid_embeddings if args.epitope_valid_embeddings else config['embeddings']['epitope_valid']

    # Load Data from WandB artifact
    dataset_name = f"beta_allele"
    artifact = wandb.use_artifact("ba_cancerimmunotherapy/dataset-allele/beta_allele:latest")
    data_dir = artifact.download(f"./WnB_Experiments_Datasets/{dataset_name}")
        
    train_file_path = f"{data_dir}/allele/train.tsv"
    val_file_path = f"{data_dir}/allele/validation.tsv"

    train_data = pd.read_csv(train_file_path, sep="\t")
    val_data = pd.read_csv(val_file_path, sep="\t")

    physchem_map = pd.read_csv("../../../../data/physico/descriptor_encoded_physchem_mapping.tsv", sep="\t")

    # Merge physicochemical data
    train_data = pd.merge(train_data, physchem_map, on=["TRB_CDR3", "Epitope"], how="left")
    val_data = pd.merge(val_data, physchem_map, on=["TRB_CDR3", "Epitope"], how="left")

    # Create mappings from training data
    trbv_dict = {v: i for i, v in enumerate(train_data["TRBV"].unique())}
    trbj_dict = {v: i for i, v in enumerate(train_data["TRBJ"].unique())}
    mhc_dict  = {v: i for i, v in enumerate(train_data["MHC"].unique())}

    UNKNOWN_TRBV_IDX = len(trbv_dict)
    UNKNOWN_TRBJ_IDX = len(trbj_dict)
    UNKNOWN_MHC_IDX  = len(mhc_dict)

    # Apply mappings to both datasets
    for df in [train_data, val_data]:
        df["TRBV_Index"] = df["TRBV"].map(trbv_dict).fillna(UNKNOWN_TRBV_IDX).astype(int)
        df["TRBJ_Index"] = df["TRBJ"].map(trbj_dict).fillna(UNKNOWN_TRBJ_IDX).astype(int)
        df["MHC_Index"]  = df["MHC"].map(mhc_dict).fillna(UNKNOWN_MHC_IDX).astype(int)

    # Vocabulary sizes
    trbv_vocab_size = UNKNOWN_TRBV_IDX + 1
    trbj_vocab_size = UNKNOWN_TRBJ_IDX + 1
    mhc_vocab_size  = UNKNOWN_MHC_IDX + 1

    print(f"Vocabulary sizes - TRBV: {trbv_vocab_size}, TRBJ: {trbj_vocab_size}, MHC: {mhc_vocab_size}")

    # Check class distribution
    pos_count = (train_data['Binding'] == 1).sum()
    neg_count = (train_data['Binding'] == 0).sum()
    print(f"Training class distribution - Positive: {pos_count}, Negative: {neg_count}")
    print(f"Class imbalance ratio: {neg_count/pos_count:.2f}:1")

    # Load Embeddings with lazy loading
    def load_h5_lazy(file_path):
        return h5py.File(file_path, 'r')

    print('Loading embeddings...')
    tcr_train_embeddings = load_h5_lazy(tcr_train_path)
    epitope_train_embeddings = load_h5_lazy(epitope_train_path)
    tcr_valid_embeddings = load_h5_lazy(tcr_valid_path)
    epitope_valid_embeddings = load_h5_lazy(epitope_valid_path)

    with h5py.File(config['embeddings']['physchem'], 'r') as f:
        inferred_physchem_dim = f["tcr_encoded"].shape[1]

    # Create datasets
    train_dataset = LazyFullFeatureDataset(train_data, tcr_train_embeddings, epitope_train_embeddings, physchem_file, trbv_dict, trbj_dict, mhc_dict)
    val_dataset = LazyFullFeatureDataset(val_data, tcr_valid_embeddings, epitope_valid_embeddings, physchem_file, trbv_dict, trbj_dict, mhc_dict)

    # Create balanced sampler for training
    train_labels = train_data['Binding'].values.astype(int)
    balanced_sampler = create_balanced_sampler(train_labels, batch_size)

    # Data loaders - using balanced sampler instead of rotating sampler
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=balanced_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    dropout = args.dropout if args.dropout else config.get('dropout', 0.5)  # Increased default dropout
    print(f"Using dropout: {dropout}")

    model = TCR_Epitope_Transformer_AllFeatures(
        config['embed_dim'],
        config['num_heads'],
        config['num_layers'],
        config['max_tcr_length'],
        config['max_epitope_length'],
        dropout=dropout,
        physchem_dim=inferred_physchem_dim,
        classifier_hidden_dim=config.get('classifier_hidden_dim', 64),
        trbv_vocab_size=trbv_vocab_size,
        trbj_vocab_size=trbj_vocab_size,
        mhc_vocab_size=mhc_vocab_size 
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    wandb.watch(model, log="all", log_freq=100)

    # Loss function with class weighting
    pos_weight = torch.tensor([neg_count / pos_count]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Using pos_weight: {pos_weight.item():.3f}")

    # Optimizer configuration from wandb config or args
    # Ensure weight_decay has a valid value

    learning_rate = args.learning_rate if args.learning_rate else wandb.config.get('learning_rate', config.get('learning_rate', 1e-4))
    optimizer_name = args.optimizer if hasattr(args, 'optimizer') else wandb.config.get("optimizer", config.get("optimizer", "adam"))
    weight_decay = args.weight_decay if hasattr(args, 'weight_decay') else wandb.config.get("weight_decay", config.get("weight_decay", 1e-4))  # Increased default weight decay
    if weight_decay is None:
        weight_decay = 0.0
        
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    print(f"Using optimizer: {optimizer_name}, lr: {learning_rate}, weight_decay: {weight_decay}")

    # Learning rate scheduler - more aggressive
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True, min_lr=1e-6)

    # Early stopping setup
    early_stopping = EarlyStopping(patience=4, min_delta=0.001, restore_best_weights=True)

    # Training tracking
    best_f1 = 0.0
    best_model_state = None
    global_step = 0

    print("Starting improved training...")

    # Training Loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False)

        for tcr, epitope, tcr_phys, epi_phys, trbv, trbj, mhc, label in train_loader_tqdm:
            tcr, epitope, tcr_phys, epi_phys, trbv, trbj, mhc, label = (
                tcr.to(device), epitope.to(device), tcr_phys.to(device), epi_phys.to(device),
                trbv.to(device), trbj.to(device), mhc.to(device), label.to(device)
            )
            
            optimizer.zero_grad()
            output = model(tcr, epitope, tcr_phys, epi_phys, trbv, trbj, mhc)
            loss = criterion(output, label)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
            
            wandb.log({"train_loss": loss.item(), "epoch": epoch + 1}, step=global_step)
            global_step += 1

            train_loader_tqdm.set_postfix(loss=epoch_loss / num_batches)

        avg_train_loss = epoch_loss / num_batches

        # Validation
        model.eval()
        all_labels = []
        all_outputs = []
        val_loss_total = 0

        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]", leave=False)

        with torch.no_grad():
            for tcr, epitope, tcr_phys, epi_phys, trbv, trbj, mhc, label in val_loader_tqdm:
                tcr, epitope, tcr_phys, epi_phys, trbv, trbj, mhc, label = (
                    tcr.to(device), epitope.to(device), tcr_phys.to(device), epi_phys.to(device),
                    trbv.to(device), trbj.to(device), mhc.to(device), label.to(device)
                )
                
                output = model(tcr, epitope, tcr_phys, epi_phys, trbv, trbj, mhc)
                val_loss = criterion(output, label)
                val_loss_total += val_loss.item()

                probs = torch.sigmoid(output)
                all_labels.extend(label.cpu().numpy())
                all_outputs.extend(probs.cpu().numpy())

        all_labels = np.array(all_labels)
        all_outputs = np.array(all_outputs)

        # Evaluate with optimal threshold
        val_metrics, val_preds = evaluate_with_optimal_threshold(all_labels, all_outputs)
        
        current_lr = optimizer.param_groups[0]['lr']
        avg_val_loss = val_loss_total / len(val_loader)

        # Print epoch results
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.2e}")
        print(f"Optimal Threshold: {val_metrics['threshold']:.3f}")
        print(f"Val F1: {val_metrics['f1']:.4f}, Val AUC: {val_metrics['auc']:.4f}, Val AP: {val_metrics['ap']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
        print(f"TP: {val_metrics['tp']}, TN: {val_metrics['tn']}, FP: {val_metrics['fp']}, FN: {val_metrics['fn']}")

        # ROC Curve plotting
        if val_metrics['auc'] > 0:
            fpr, tpr, _ = roc_curve(all_labels, all_outputs)
            
            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {val_metrics["auc"]:.3f})')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            
            os.makedirs("results", exist_ok=True)
            roc_curve_path = f"results/roc_curve_epoch_{epoch+1}.png"
            plt.savefig(roc_curve_path)
            wandb.log({"roc_curve": wandb.Image(roc_curve_path)}, step=global_step, commit=False)
            plt.close()

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss_epoch": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_auc": val_metrics['auc'],
            "val_ap": val_metrics['ap'],
            "val_f1": val_metrics['f1'],
            "val_accuracy": val_metrics['accuracy'],
            "val_precision": val_metrics['precision'],
            "val_recall": val_metrics['recall'],
            "optimal_threshold": val_metrics['threshold'],
            "val_tp": val_metrics['tp'],
            "val_tn": val_metrics['tn'],
            "val_fp": val_metrics['fp'],
            "val_fn": val_metrics['fn'],
            "learning_rate": current_lr,
            "prediction_distribution": wandb.Histogram(all_outputs),
            "label_distribution": wandb.Histogram(all_labels),
            "val_confusion_matrix": wandb.plot.confusion_matrix(
                y_true=all_labels.astype(int),
                preds=val_preds.astype(int),
                class_names=["Not Binding", "Binding"])
        }, step=global_step, commit=False)

        # TPP1–TPP4 Evaluation
        if "task" in val_data.columns:
            all_tasks = val_data["task"].values

            for tpp in ["TPP1", "TPP2", "TPP3", "TPP4"]:
                mask = all_tasks == tpp
                if mask.sum() > 0:
                    tpp_labels = all_labels[mask]
                    tpp_outputs = all_outputs[mask]
                    
                    # Evaluate TPP with optimal threshold
                    tpp_metrics, tpp_preds = evaluate_with_optimal_threshold(tpp_labels, tpp_outputs)

                    print(f"\n    {tpp} ({mask.sum()} samples)")
                    print(f"    AUC: {tpp_metrics['auc']:.4f}, AP: {tpp_metrics['ap']:.4f}")
                    print(f"    F1: {tpp_metrics['f1']:.4f}, Acc: {tpp_metrics['accuracy']:.4f}")
                    print(f"    Precision: {tpp_metrics['precision']:.4f}, Recall: {tpp_metrics['recall']:.4f}")
                    print(f"    TP: {tpp_metrics['tp']}, TN: {tpp_metrics['tn']}, FP: {tpp_metrics['fp']}, FN: {tpp_metrics['fn']}")

                    # Log TPP metrics
                    tpp_log_dict = {
                        f"val_{tpp}_f1": tpp_metrics['f1'],
                        f"val_{tpp}_accuracy": tpp_metrics['accuracy'],
                        f"val_{tpp}_precision": tpp_metrics['precision'],
                        f"val_{tpp}_recall": tpp_metrics['recall'],
                        f"val_{tpp}_threshold": tpp_metrics['threshold'],
                        f"val_{tpp}_tp": tpp_metrics['tp'],
                        f"val_{tpp}_tn": tpp_metrics['tn'],
                        f"val_{tpp}_fp": tpp_metrics['fp'],
                        f"val_{tpp}_fn": tpp_metrics['fn']
                    }
                    
                    if tpp_metrics['auc'] > 0:
                        tpp_log_dict[f"val_{tpp}_auc"] = tpp_metrics['auc']
                    if tpp_metrics['ap'] > 0:
                        tpp_log_dict[f"val_{tpp}_ap"] = tpp_metrics['ap']

                    wandb.log(tpp_log_dict, step=global_step, commit=False)

                    # TPP Confusion Matrix
                    wandb.log({
                        f"val_{tpp}_confusion_matrix": wandb.plot.confusion_matrix(
                            y_true=tpp_labels.astype(int),
                            preds=tpp_preds.astype(int),
                            class_names=["Not Binding", "Binding"],
                            title=f"Confusion Matrix – {tpp}"
                        )
                    }, step=global_step, commit=False)

                    # TPP Prediction Distribution
                    plt.figure(figsize=(6, 4))
                    plt.hist(tpp_outputs, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
                    plt.title(f"Prediction Distribution – {tpp}")
                    plt.xlabel("Predicted Probability")
                    plt.ylabel("Frequency")
                    plt.axvline(tpp_metrics['threshold'], color='red', linestyle='--', label=f'Optimal Threshold: {tpp_metrics["threshold"]:.3f}')
                    plt.legend()
                    plt.tight_layout()
                    
                    plot_path = f"results/{tpp}_pred_dist_epoch_{epoch+1}.png"
                    os.makedirs("results", exist_ok=True)
                    plt.savefig(plot_path)
                    wandb.log({f"val_{tpp}_prediction_distribution": wandb.Image(plot_path)}, step=global_step, commit=False)
                    plt.close()

        # Learning rate scheduling based on F1-score
        scheduler.step(val_metrics['f1'])

        # Early stopping check (based on F1-score now)
        if early_stopping(val_metrics['f1'], model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            early_stopping.restore_best_weights_to_model(model)
            print("Restored best weights")
            break

        # Track best model based on F1-score
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Save checkpoint every epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_f1': val_metrics['f1'],
            'val_auc': val_metrics['auc'],
            'optimal_threshold': val_metrics['threshold'],
            'config': config
        }
        
        checkpoint_path = f"model_checkpoint_epoch_{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Log model checkpoint to wandb
        artifact = wandb.Artifact(f"model_checkpoint_epoch_{epoch+1}", type="model")
        artifact.add_file(checkpoint_path)
        run.log_artifact(artifact)
        wandb.log({}, step=global_step, commit=True)  # Commit all logs for this epoch

    # Save best model
    if best_model_state:
        # Load best weights into model
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(best_model_state, model_path)
        print(f"Best model saved with F1: {best_f1:.4f}")

        # Log final best model artifact
        final_artifact = wandb.Artifact(run_name + "_best_model", type="model")
        final_artifact.add_file(model_path)
        wandb.log_artifact(final_artifact)

    print("Training completed!")
    print(f"Best validation F1: {best_f1:.4f}")
    wandb.finish()

if __name__ == "__main__":
    main()