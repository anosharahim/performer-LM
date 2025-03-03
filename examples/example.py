import os
import sys
import torch
import time
import json
from tqdm import tqdm  # for progress bar
from pathlib import Path
import matplotlib.pyplot as plt


os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.append(os.getcwd()) 

from performer.model import Transformer
from performer.dataset import WikiText103Dataset
from performer.train import train_model, plot_loss, plot_lr
from performer.tokenizer import create_tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "cpu"
)
print(f"Using device: {device}")

embedding_dim = 64 # reduce from 128
num_heads = 4 # reduce from 8
ff_dim = embedding_dim * 4
num_layers = 2 # reduce from 6
dropout_rate = 0.3

seq_len = 50 # reduce from 100
batch_size = 16 # reduce from 32
num_epochs = 50
learning_rate = 0.005
num_samples = 500

initial_vocab_size = 5000 
max_samples = 50000 
val_split = 0.1 
print(f"Using a subset of {max_samples} samples from a total of ...")

data_dir = Path("./data")
data_dir.mkdir(exist_ok=True)

dataset_cache_path = data_dir / "wikitext103_dataset.pt"
tokenizer_path = data_dir / "tokenizer.json"

# Dataset caching 
if dataset_cache_path.exists():
    print("Loading cached dataset...")
    dataset = torch.load(dataset_cache_path)
else:
    print("Downloading dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    torch.save(dataset, dataset_cache_path)
    print("Dataset cached for future use.")

# Get total number of samples in the training set
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

# Extract actual vocabulary size from tokenizer
vocab_size = tokenizer.get_vocab_size()
print(f"Using vocabulary size: {vocab_size}")

data_subset = dataset['train'].select(range(max_samples))

train_size = int(len(data_subset) * (1 - val_split))
val_size = len(data_subset) - train_size
print(f"Splitting into {train_size} training samples and {val_size} validation samples")

train_subset = data_subset.select(range(train_size))
val_subset = data_subset.select(range(train_size, len(data_subset)))

train_dataset = WikiText103Dataset(train_subset, tokenizer, seq_len)
val_dataset = WikiText103Dataset(val_subset, tokenizer, seq_len)

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
    vocab_size=vocab_size
).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=.01)

training_results = train_model(
    model, 
    train_dataloader, 
    criterion, 
    optimizer, 
    num_epochs=num_epochs, 
    device=device,
    val_dataloader=val_dataloader
)

# Extract training and validation losses
train_loss_history = training_results['train_loss']
val_loss_history = training_results['val_loss']
lr_history = training_results['lr']

# Plot losses with both training and validation
plot_loss(train_loss_history, val_loss_history)
plot_lr(lr_history)

# Print final losses
print(f"Final training loss: {train_loss_history[-1]:.4f}")
print(f"Final validation loss: {val_loss_history[-1]:.4f}")

# Check for overfitting
if val_loss_history[-1] > train_loss_history[-1] * 1.1:  # 10% higher validation loss indicates overfitting
    print("Warning: Model may be overfitting (validation loss > training loss).")
    # Calculate where validation loss starts diverging from training loss
    for epoch in range(1, len(val_loss_history)):
        if val_loss_history[epoch] > val_loss_history[epoch-1] and train_loss_history[epoch] < train_loss_history[epoch-1]:
            print(f"Overfitting likely began around epoch {epoch+1}")
            break
