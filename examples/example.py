import os
import sys
import torch
import time
from tqdm import tqdm  # for progress bar


os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.append(os.getcwd()) 

from performer.model import Transformer
from performer.dataset import WikiText103Dataset
from performer.train import train_model
from performer.tokenizer import create_tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "cpu"
)
print (f"Using device: {device}")

embedding_dim = 128
num_heads = 8
ff_dim = embedding_dim * 4
num_layers = 6
dropout_rate = 0.1

seq_len = 100
batch_size = 32
num_epochs = 1
learning_rate = 0.005 #implement lr scheduling
num_samples = 10

dataset = load_dataset("wikitext", "wikitext-103-v1")
vocab_size = 8000  # Standard size for many language models, adjust if needed
tokenizer, trainer = create_tokenizer(vocab_size)

# Train the tokenizer on the dataset
def get_training_corpus():
    for i in range(0, len(dataset['train']), num_samples):
        yield dataset['train'][i:i + num_samples]['text']

tokenizer.train_from_iterator(get_training_corpus(), trainer)

# Get actual vocabulary size after training
# vocab_size = tokenizer.get_vocab_size()
# print(f"Actual vocabulary size after training: {vocab_size}")

train_dataset = WikiText103Dataset(dataset['train'], tokenizer, seq_len)
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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

train_model(model, dataloader, criterion, optimizer, num_epochs, vocab_size, device)
