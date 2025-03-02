import torch
import matplotlib.pyplot as plt
import os

def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.train()
    model.to(device)
    
    vocab_size = model.linear.out_features
    
    loss_history = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_source, batch_target in dataloader:
            batch_source, batch_target = batch_source.to(device), batch_target.to(device)

            optimizer.zero_grad()
            output = model(batch_source, batch_target[:, :-1])
            output = output.reshape(-1, vocab_size)
            target = batch_target[:, 1:].reshape(-1)
            loss = criterion(output, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #gradient clipping
            optimizer.step()

            total_loss += loss.item()
            
        avg_loss = total_loss/len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    return loss_history

def plot_loss(loss_history, save_dir='data/loss_graphs', filename='loss_curve.png', overwrite=False):
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, filename)
    
    if not overwrite and os.path.exists(save_path):
        base, ext = os.path.splitext(filename)
        i = 1
        while os.path.exists(os.path.join(save_dir, f"{base}_{i}{ext}")):
            i += 1
        save_path = os.path.join(save_dir, f"{base}_{i}{ext}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to {save_path}")



