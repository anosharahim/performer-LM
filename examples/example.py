import os
import sys
import torch
import time
import json
from tqdm import tqdm  # for progress bar
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.append(os.getcwd()) 

from performer.model import Transformer
from performer.dataset import WikiText103Dataset
from performer.train import train_model, plot_loss
from performer.tokenizer import create_tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "cpu"
)
print(f"Using device: {device}")

embedding_dim = 64 
num_heads = 2
ff_dim = embedding_dim * 4
num_layers = 2 
dropout_rate = 0.3 

seq_len = 128  
batch_size = 16 
num_epochs = 30
learning_rate = 0.0008 
weight_decay = 1e-2  
num_samples = 500

initial_vocab_size = 10000  
max_samples = 10000
val_split = 0.2
print(f"Using a subset of {max_samples} samples from a total of ...")

data_dir = Path("./data")
data_dir.mkdir(exist_ok=True)

dataset_cache_path = data_dir / "wikitext103_dataset.pt"
tokenizer_path = data_dir / "tokenizer.json"
model_save_path = data_dir / "trained_model.pt"  # Path to save the model

# Dataset caching 
if dataset_cache_path.exists():
    print("Loading cached dataset...")
    dataset = torch.load(dataset_cache_path)
else:
    print("Downloading dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    torch.save(dataset, dataset_cache_path)
    print("Dataset cached for future use.")

total_samples = len(dataset['train'])

# Check for existing tokenizer
if tokenizer_path.exists():
    print("Loading pre-trained tokenizer...")
    tokenizer, _ = create_tokenizer(initial_vocab_size, load_path=str(tokenizer_path))
else:
    print("Training tokenizer...")
    tokenizer, trainer = create_tokenizer(initial_vocab_size)
    
    # train+save tokenizer on dataset
    def get_training_corpus():
        for i in range(0, len(dataset['train']), num_samples):
            yield dataset['train'][i:i + num_samples]['text']
    
    tokenizer.train_from_iterator(get_training_corpus(), trainer)
    tokenizer.save(str(tokenizer_path))
    print("Tokenizer saved for future use.")

vocab_size = tokenizer.get_vocab_size()
print(f"Using vocabulary size: {vocab_size}")

data_subset = dataset['train'].select(range(max_samples))
data_subset = data_subset.select(np.random.permutation(len(data_subset)).tolist())  # Shuffle to prevent data leakage

train_size = int(len(data_subset) * (1 - val_split))
val_size = len(data_subset) - train_size
print(f"Splitting into {train_size} training samples and {val_size} validation samples")

train_subset = data_subset.select(range(train_size))
val_subset = data_subset.select(range(train_size, len(data_subset)))

train_dataset = WikiText103Dataset(train_subset, tokenizer, seq_len)
val_dataset = WikiText103Dataset(val_subset, tokenizer, seq_len)

# Data leakage check: Verify train and validation sets don't overlap
train_texts_dict = {}
val_texts_dict = {}

# Only consider non-empty samples with substantial content (>20 chars)
for i in range(min(500, len(train_subset))):
    text = train_subset[i]['text'][:150].strip()
    if len(text) > 20 and not text.startswith('= ='):  # Skip empty/headers
        train_texts_dict[text] = i
        
for i in range(min(500, len(val_subset))):
    text = val_subset[i]['text'][:150].strip()
    if len(text) > 20 and not text.startswith('= ='):  # Skip empty/headers
        val_texts_dict[text] = i

overlapping_keys = set(train_texts_dict.keys()) & set(val_texts_dict.keys())
if overlapping_keys:
    print(f"\n⚠️ WARNING: DATA LEAKAGE DETECTED! {len(overlapping_keys)} samples appear in both train and validation sets.")
    
    # Print details of up to 3 overlapping samples
    for i, key in enumerate(list(overlapping_keys)[:3]):
        train_idx = train_texts_dict[key]
        val_idx = val_texts_dict[key]
        print(f"\n--- Overlapping Sample #{i+1} ---")
        print(f"Train idx: {train_idx}, Val idx: {val_idx}")
        sample_text = key.strip()
        if sample_text:
            print(f"Content: \"{sample_text[:100]}...\"")
        else:
            print("Content: [Empty string]")
    
    # Check for sequential text patterns
    print("\n--- Analyzing Sequential Text Patterns ---")
    for i in range(min(3, len(train_subset))):
        train_end = train_subset[train_size - i - 1]['text'][-50:].strip()
        val_start = val_subset[0]['text'][:50].strip()
        
        if not train_end or not val_start:
            continue
            
        print(f"\nChecking sample boundaries {train_size - i - 1} → {train_size}:")
        print(f"Train end:  \"{train_end}\"")
        print(f"Val start:  \"{val_start}\"")
        
        # Count matching characters at boundary
        similarity = sum(1 for a, b in zip(train_end, val_start) if a == b)
        similarity_pct = similarity / min(len(train_end), len(val_start)) * 100
        print(f"Similarity: {similarity_pct:.1f}% ({similarity} matching chars)")
        
        if similarity_pct > 50:
            print("→ Sequential text detected between train and validation splits!")
    
    print("\n❌ TRAINING ABORTED: Data leakage would lead to unreliable evaluation metrics.")
    print("Fix by adding the random shuffle step before splitting the dataset:")
    print("  shuffle_indices = np.random.permutation(len(data_subset))")
    print("  data_subset = data_subset.select(shuffle_indices.tolist())")
    sys.exit(1)
else:
    print("✅ No data leakage detected in samples checked.")

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Using a subset of {max_samples} samples from a total of {total_samples} samples")
print(f"Total number of training batches: {len(train_dataloader)}")
print(f"Total number of validation batches: {len(val_dataloader)}")

model = Transformer(
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    dropout_rate=dropout_rate,
    num_layers=num_layers,
    vocab_size=vocab_size,
    encoder_attention_type='fast_attention',
    num_random_features=embedding_dim / num_heads,
).to(device) 

print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

training_results = train_model(
    model, 
    train_dataloader, 
    criterion, 
    optimizer, 
    num_epochs=num_epochs, 
    device=device,
    val_dataloader=val_dataloader
)

#training + validation loss 
train_loss_history = training_results['train_loss']
val_loss_history = training_results['val_loss']
val_ppl_history = training_results['val_ppl']
lr_history = training_results['lr']
total_time = training_results['total_time']

plot_loss(train_loss_history, val_loss_history, val_ppl_history)

# Print final losses
print(f"Final training loss: {train_loss_history[-1]:.4f}")
print(f"Final validation loss: {val_loss_history[-1]:.4f}")
print(f"Total training time: {total_time:.2f} seconds")

# Check for overfitting: Calculate where validation loss starts diverging from training loss
if val_loss_history[-1] > train_loss_history[-1] * 1.1:
    print("Warning: Model may be overfitting (validation loss > training loss).")
    
    for epoch in range(1, len(val_loss_history)):
        if val_loss_history[epoch] > val_loss_history[epoch-1] and train_loss_history[epoch] < train_loss_history[epoch-1]:
            print(f"Overfitting likely began around epoch {epoch+1}")
            break

# Save the trained model
print(f"Saving trained model to {model_save_path}...")
torch.save(model.state_dict(), model_save_path)
print(f"Model saved successfully!")

# Save model configuration for future reference
model_config = {
    "embedding_dim": embedding_dim,
    "num_heads": num_heads,
    "ff_dim": ff_dim, 
    "dropout_rate": dropout_rate,
    "num_layers": num_layers,
    "vocab_size": vocab_size,
}
config_path = data_dir / "model_config.json"
with open(config_path, 'w') as f:
    json.dump(model_config, f, indent=4)
print(f"Model configuration saved to {config_path}")

print("\nTo run inference with this model, use the following command:")
print(f"python examples/inference_example.py --model_path={model_save_path} --tokenizer_path={tokenizer_path}")
