import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for TCR-Epitope prediction model")
    
    # Paths
    parser.add_argument('--configs_path', type=str, default='/configs/v1_mha_config.yaml', help='Path to configuration file')
    
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
