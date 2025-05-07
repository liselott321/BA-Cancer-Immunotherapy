import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for TCR-Epitope prediction model")
    
    # Paths
    parser.add_argument('--configs_path', type=str, default='./configs/v1_mha_1024_config-serv1.yaml', help='Path to configuration file')
    parser.add_argument('--train', type=str, help='Path to training dataset')
    parser.add_argument('--val', type=str, help='Path to validation dataset')
    parser.add_argument('--test', type=str, help='Path to testing dataset')
    parser.add_argument('--tcr_embeddings', type=str, help='Path to TCR embeddings')
    parser.add_argument('--epitope_embeddings', type=str, help='Path to Epitope embeddings')
    parser.add_argument('--model_path', type=str, help='Path to save best model')
    parser.add_argument('--tcr_train_embeddings', type=str, help='Path to train TCR embeddings')
    parser.add_argument('--epitope_train_embeddings', type=str, help='Path to train Epitope embeddings')
    parser.add_argument('--tcr_test_embeddings', type=str, help='Path to test TCR embeddings')
    parser.add_argument('--epitope_test_embeddings', type=str, help='Path to test Epitope embeddings')
    parser.add_argument('--tcr_valid_embeddings', type=str, help='Path to valid TCR embeddings')
    parser.add_argument('--epitope_valid_embeddings', type=str, help='Path to valid Epitope embeddings')
    parser.add_argument('--ple_path', type=str, default=None, help='Pfad zur PLE HDF5 Datei')
    parser.add_argument("--descriptor_path", type=str, help="Pfad zu den globalen Deskriptor-Features (HDF5)")
    parser.add_argument('--artifact_name', type=str, help='Path to best model to test')

    
    # Training Parameters
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], help='Optimizer to use')
    parser.add_argument('--weight_decay', type=float, help='Weight decay value for optimizer')
    parser.add_argument('--dropout', type=float, help='Dropout value for TCR_Epitope_transformer class')
    parser.add_argument("--penalty_weight", type=float, help="Weight for confidence penalty term")
    
    # Model Parameters
    parser.add_argument('--embed_dim', type=int, help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, help='Number of transformer layers')
    parser.add_argument('--max_tcr_length', type=int, help='Max TCR sequence length')
    parser.add_argument('--max_epitope_length', type=int, help='Max Epitope sequence length')
    parser.add_argument('--classifier_hidden_dim', type=int, help='Hidden dimension of the classification head')
    
    return parser.parse_args()
