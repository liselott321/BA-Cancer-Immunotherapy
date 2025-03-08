import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import wandb

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, output_attn_w=False, n_hidden=64, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.mh = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.drop2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, embed_dim)
        )
        self.output_attn_w = output_attn_w

    def forward(self, x):
        xm, attn_w = self.mh(x, x, x)
        xm = self.drop1(xm)
        xm = self.norm1(x + xm)
        x = self.ff(xm)
        x = self.drop2(x)
        xm = self.norm2(x + xm)
        return (xm, attn_w) if self.output_attn_w else xm

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_linear):
        super(Classifier, self).__init__()
        self.relu = nn.ReLU()
        self.downsampling_linear = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_linear)
        self.res_block1 = ResidualBlock(hidden_dim, dropout_linear)
        self.final_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.downsampling_linear(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.res_block1(x)
        x = self.final_layer(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_linear):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_linear)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.linear1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out += residual
        return out

class TCR_Epitope_Model(pl.LightningModule):
    def __init__(self, embed_dim, max_seq_length, hyperparameters):
        super(TCR_Epitope_Model, self).__init__()
        self.save_hyperparameters()
        self.auroc = BinaryAUROC(thresholds=None)
        self.avg_precision = BinaryAveragePrecision(thresholds=None)
        self.hyperparameters = hyperparameters
        self.max_seq_length = max_seq_length

        self.transformer_in = embed_dim
        self.num_heads = 4
        self.n_hidden = int(1.5 * self.transformer_in)
        self.multihead_attn_global = TransformerBlock(self.transformer_in, self.num_heads, False, self.n_hidden, self.hyperparameters["dropout_attention"])

        self.classifier_hidden = 64
        self.classifier_in = (2 * max_seq_length) * embed_dim
        self.classifier = Classifier(self.classifier_in, self.classifier_hidden, self.hyperparameters["dropout_linear"])

    def forward(self, epitope_embedding, tcr_embedding):
        combined_embedding = torch.cat([tcr_embedding, epitope_embedding], dim=1)
        combined_embedding = combined_embedding.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        attention_output = self.multihead_attn_global(combined_embedding)
        attention_output = attention_output.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim)
        flattened_output = attention_output.view(attention_output.size(0), -1)
        logits = self.classifier(flattened_output)
        return logits

    def training_step(self, batch, batch_idx):
        epitope_embedding = batch["epitope_embedding"]
        tcr_embedding = batch["tcr_embedding"]
        label = batch["label"]
        
        output = self(epitope_embedding, tcr_embedding).squeeze()
        loss = F.binary_cross_entropy_with_logits(output, label)
        self.log("train_loss", loss, on_step=True, prog_bar=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        epitope_embedding = batch["epitope_embedding"]
        tcr_embedding = batch["tcr_embedding"]
        label = batch["label"]
        
        output = self(epitope_embedding, tcr_embedding).squeeze()
        val_loss = F.binary_cross_entropy_with_logits(output, label)
        self.log("val_loss", val_loss, batch_size=len(batch))
        
        prediction = torch.sigmoid(output)
        self.log("ROCAUC_Val", self.auroc(prediction, label), on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log("AP_Val", self.avg_precision(prediction, label.to(torch.long)), on_epoch=True, prog_bar=True, batch_size=len(batch))
        return val_loss

    def configure_optimizers(self):
        optimizer = self.hyperparameters["optimizer"]
        learning_rate = self.hyperparameters["learning_rate"]
        weight_decay = self.hyperparameters["weight_decay"]
        betas = (0.9, 0.98)
        
        if optimizer == "sgd": 
            optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)        
        elif optimizer == "adam": 
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
        else: 
            print("OPTIMIZER NOT FOUND")
        
        return optimizer

    def on_validation_epoch_end(self):
        all_predictions = torch.cat(self.val_predictions).numpy()
        all_labels = torch.cat(self.val_labels).numpy()

        precision = precision_score(all_labels, all_predictions, zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_predictions)
    
        wandb.log({
            "val_precision": precision,
            "val_confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_labels,
                preds=all_predictions,
                class_names=["Not Binding", "Binding"]
            ),
        })
    
        self.log("val_precision", precision, on_epoch=True, prog_bar=True)

        positive_preds = all_predictions[all_labels == 1]
        negative_preds = all_predictions[all_labels == 0]
    
        plt.figure(figsize=(10, 6))
        plt.hist(negative_preds, bins=50, alpha=0.7, label="Negative Predictions")
        plt.hist(positive_preds, bins=50, alpha=0.7, label="Positive Predictions")
        plt.xlabel("Sigmoid Output")
        plt.ylabel("Frequency")
        plt.title("Prediction Distribution (Validation)")
        plt.legend()
        plt.show()
    
        wandb.log({"validation_prediction_histogram": plt})
    
        self.val_predictions.clear()
        self.val_labels.clear()