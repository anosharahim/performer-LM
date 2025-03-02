import torch
import matplotlib.pyplot as plt
import os
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

def train_model(model, dataloader, criterion, optimizer, num_epochs, device, 
                use_lr_scheduler=True, warmup_epochs=5, cosine_T_max=None, cosine_eta_min=1e-6):
    model.train()
    model.to(device)
    
    vocab_size = model.linear.out_features
    
    loss_history = []
    lr_history = []
    
    if use_lr_scheduler:
        def warmup_fn(epoch):
            return min(1.0, epoch / warmup_epochs)
        
        warmup_scheduler = LambdaLR(optimizer, warmup_fn)
        
        if cosine_T_max is None:
            cosine_T_max = num_epochs - warmup_epochs
        
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_T_max, eta_min=cosine_eta_min)
    
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
        
        if use_lr_scheduler:
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
            
            lr_history.append(optimizer.param_groups[0]['lr'])
            
        avg_loss = total_loss/len(dataloader)
        loss_history.append(avg_loss)
        
        if use_lr_scheduler:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    return loss_history, lr_history if use_lr_scheduler else loss_history

def plot_loss(loss_history, save_dir='data/loss_graphs', filename='loss_curve.png', overwrite=False):
    """
    Plot and save the loss curve
    """
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

def plot_lr(lr_history, save_dir='data/lr_graphs', filename='lr_curve.png', overwrite=False):
    """
    Plot and save the learning rate curve
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    full_path = os.path.join(save_dir, filename)
    
    if os.path.exists(full_path) and not overwrite:
        print(f"File {full_path} already exists. Skipping plot.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(lr_history, 'b-')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Scheduler')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(full_path)
    plt.close()
    print(f"Learning rate curve saved to {full_path}")



