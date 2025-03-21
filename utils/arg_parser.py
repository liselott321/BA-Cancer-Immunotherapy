import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for TCR-Epitope prediction model")
    
    # Paths
    parser.add_argument('--configs_path', type=str, default='./configs/v1_mha_config.yaml', help='Path to configuration file')
    parser.add_argument('--train', type=str, help='Path to training dataset')
    parser.add_argument('--val', type=str, help='Path to validation dataset')
    parser.add_argument('--tcr_embeddings', type=str, help='Path to TCR embeddings')
    parser.add_argument('--epitope_embeddings', type=str, help='Path to Epitope embeddings')
    parser.add_argument('--model_path', type=str, help='Path to save best model')
    parser.add_argument('--tcr_train_embeddings', type=str, help='Path to train TCR embeddings')
    parser.add_argument('--epitope_train_embeddings', type=str, help='Path to train Epitope embeddings')
    parser.add_argument('--tcr_valid_embeddings', type=str, help='Path to valid TCR embeddings')
    parser.add_argument('--epitope_valid_embeddings', type=str, help='Path to valid Epitope embeddings')

    
    # Training Parameters
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    
    # Model Parameters
    parser.add_argument('--embed_dim', type=int, help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, help='Number of transformer layers')
    parser.add_argument('--max_tcr_length', type=int, help='Max TCR sequence length')
    parser.add_argument('--max_epitope_length', type=int, help='Max Epitope sequence length')
    
    return parser.parse_args()
