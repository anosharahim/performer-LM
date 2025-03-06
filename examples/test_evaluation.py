import os
import sys
import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.append(os.getcwd())

from performer.model import Transformer
from performer.dataset import WikiText103Dataset
from performer.train import evaluate_model
from performer.tokenizer import create_tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# Configuration - Same as your training script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Directories and paths
data_dir = Path("./data")
dataset_cache_path = data_dir / "wikitext103_dataset.pt"
tokenizer_path = data_dir / "tokenizer.json"
model_path = data_dir / "trained_model.pt"  # Path to your trained model
config_path = data_dir / "model_config.json"

# Parameters
batch_size = 32
seq_len = 500
test_start_idx = 15000  # Start index for test set (completely held-out)
test_size = 2000  # Size of test set

# Load dataset
if dataset_cache_path.exists():
    print("Loading cached dataset...")
    dataset = torch.load(dataset_cache_path)
else:
    print("Downloading dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    torch.save(dataset, dataset_cache_path)

# Load model config
if not config_path.exists():
    print("Error: Model config not found. Please train a model first.")
    sys.exit(1)

with open(config_path, 'r') as f:
    model_config = json.load(f)

# Load tokenizer
if not tokenizer_path.exists():
    print("Error: Tokenizer not found. Please train a model first.")
    sys.exit(1)

print("Loading tokenizer...")
tokenizer, _ = create_tokenizer(model_config['vocab_size'], load_path=str(tokenizer_path))

# Create model with the same architecture
model = Transformer(
    embedding_dim=model_config['embedding_dim'],
    num_heads=model_config['num_heads'],
    ff_dim=model_config['ff_dim'],
    dropout_rate=model_config['dropout_rate'],
    num_layers=model_config['num_layers'],
    vocab_size=model_config['vocab_size'],
    encoder_attention_type='fast_attention',
    num_random_features=model_config['embedding_dim'] / model_config['num_heads'],
).to(device)

# Load trained weights
if not model_path.exists():
    print("Error: Trained model weights not found.")
    sys.exit(1)

print(f"Loading trained model from {model_path}...")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Create the test dataset from completely held-out portion
print(f"\n===== Evaluating on Completely Held-Out Test Set =====")
print(f"Using samples {test_start_idx} to {test_start_idx + test_size} (not used during training/validation)")

test_subset = dataset['train'].select(range(test_start_idx, test_start_idx + test_size))
test_dataset = WikiText103Dataset(test_subset, tokenizer, seq_len)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"Test set size: {len(test_dataset)} samples ({len(test_dataloader)} batches)")

# Run evaluation
criterion = torch.nn.CrossEntropyLoss()
test_loss, test_ppl = evaluate_model(model, test_dataloader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Perplexity: {test_ppl:.2f}")

# Load validation results for comparison if available
results_path = data_dir / "test_results.json"
val_metrics_path = data_dir / "val_metrics.json"

# Check if we have validation metrics to compare against
has_val_metrics = False

#get validation metrics from val_metrics.json if it exists
if val_metrics_path.exists():
    try:
        with open(val_metrics_path, 'r') as f:
            val_metrics = json.load(f)
            val_ppl = val_metrics.get('final_val_perplexity')
            if val_ppl is not None:
                has_val_metrics = True
                print("\nComparison with validation metrics:")
                print(f"Validation Perplexity: {val_ppl:.2f}")
                print(f"Test Perplexity: {test_ppl:.2f}")
                diff = abs(test_ppl - val_ppl)
                print(f"Difference: {diff:.2f} PPL points")
                
                if test_ppl > val_ppl * 1.1:
                    print("⚠️ Warning: Test perplexity is significantly higher than validation perplexity.")
                else:
                    print("✅ Test and validation metrics are reasonably close, indicating good generalization.")
    except:
        pass

if not has_val_metrics:
    print("\nNo validation metrics found for comparison.")
    print("To compare with validation metrics, run the training script first,")
    print("or manually create a val_metrics.json file with 'final_val_perplexity'.")

#save
test_results = {
    "test_loss": test_loss,
    "test_perplexity": test_ppl,
    "test_samples": test_size,
    "test_start_idx": test_start_idx
}
with open(results_path, 'w') as f:
    json.dump(test_results, f, indent=4)
print(f"Test results saved to {results_path}") 