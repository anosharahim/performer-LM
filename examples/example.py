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
from performer.train import train_model, plot_loss
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
dropout_rate = 0.1

seq_len = 50 # reduce from 100
batch_size = 16 # reduce from 32
num_epochs = 20
learning_rate = 0.005
num_samples = 10
vocab_size = 5000 # reduce from 8000

max_samples = 1000 

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

# Check for existing tokenizer
if tokenizer_path.exists():
    print("Loading pre-trained tokenizer...")
    tokenizer, _ = create_tokenizer(vocab_size, load_path=str(tokenizer_path))
else:
    print("Training tokenizer...")
    tokenizer, trainer = create_tokenizer(vocab_size)
    
    # train+save tokenizer on dataset
    def get_training_corpus():
        for i in range(0, len(dataset['train']), num_samples):
            yield dataset['train'][i:i + num_samples]['text']
    
    tokenizer.train_from_iterator(get_training_corpus(), trainer)
    tokenizer.save(str(tokenizer_path))
    print("Tokenizer saved for future use.")

print(f"Using a subset of {max_samples} samples for faster training")
data_subset = dataset['train'].select(range(max_samples))
train_dataset = WikiText103Dataset(data_subset, tokenizer, seq_len)
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print(f"Total number of batches: {len(dataloader)}")

model = Transformer(
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    dropout_rate=dropout_rate,
    num_layers=num_layers,
    vocab_size=vocab_size
).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_history = train_model(model, dataloader, criterion, optimizer, num_epochs, vocab_size, device)

plot_loss(loss_history, save_dir='data/loss_graphs')