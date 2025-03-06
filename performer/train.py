import torch
import matplotlib.pyplot as plt
import os
import time
import json
from datetime import timedelta
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

def evaluate_model(model, dataloader, criterion, device):
    '''
    Evaluates the model on the validation set during training runs.
    Returns the average loss and perplexity.
    '''
    model.eval() 
    total_loss = 0
    
    with torch.no_grad(): #disables gradient calculation
        for batch_source, batch_target in dataloader:
            batch_source, batch_target = batch_source.to(device), batch_target.to(device)
            
            vocab_size = model.linear.out_features
            output = model(batch_source, batch_target[:, :-1])
            output = output.reshape(-1, vocab_size)
            target = batch_target[:, 1:].reshape(-1)
            loss = criterion(output, target)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    model.train() #reset to training mode
    return avg_loss, perplexity

def train_model(model, train_dataloader, criterion, optimizer, num_epochs, device, 
                val_dataloader=None, use_lr_scheduler=True, warmup_epochs=10, 
                cosine_T_max=None, cosine_eta_min=1e-6):
    '''
    Handles the training loop.
    '''

    model.train()
    model.to(device)
    
    vocab_size = model.linear.out_features
    
    train_loss_history = []
    val_loss_history = []
    val_ppl_history = []
    lr_history = []
    
    start_time = time.time()
    
    if use_lr_scheduler:
        # Implements a learning rate scheduler with warmup and cosine annealing.
        def warmup_fn(epoch): 
            return min(1.0, epoch / warmup_epochs)
        warmup_scheduler = LambdaLR(optimizer, warmup_fn)
        
        if cosine_T_max is None:
            cosine_T_max = num_epochs - warmup_epochs 
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_T_max, eta_min=cosine_eta_min)
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_source, batch_target in train_dataloader:
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
            
        avg_train_loss = total_loss/len(train_dataloader)
        train_loss_history.append(avg_train_loss)
        
        # Evaluate on validation data if provided
        if val_dataloader is not None:
            val_loss, val_ppl = evaluate_model(model, val_dataloader, criterion, device)
            val_loss_history.append(val_loss)
            val_ppl_history.append(val_ppl)
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}", end="")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}", end="")
            
        if use_lr_scheduler:
            print(f", LR: {optimizer.param_groups[0]['lr']:.6f}")
        else:
            print("")
    
    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    print(f"\nTraining completed in {total_time_str}")
    
    result = {
        'train_loss': train_loss_history,
        'val_loss': val_loss_history if val_dataloader else None,
        'val_ppl': val_ppl_history if val_dataloader else None,
        'lr': lr_history if use_lr_scheduler else None,
        'total_time': total_time
    }
    
    return result

def save_loss_history(loss_data, save_dir='data/loss_histories', filename='loss_history.json', overwrite=False):
    """
    Save the loss history data to a JSON file for later access and plotting.
    
    Args:
        loss_data (dict): Dictionary containing loss histories
        save_dir (str): Directory to save the loss history file
        filename (str): Name of the JSON file
        overwrite (bool): Whether to overwrite existing files
        
    Returns:
        str: Path to the saved file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, filename)
    
    if not overwrite and os.path.exists(save_path):
        base, ext = os.path.splitext(filename)
        i = 1
        while os.path.exists(os.path.join(save_dir, f"{base}_{i}{ext}")):
            i += 1
        save_path = os.path.join(save_dir, f"{base}_{i}{ext}")
    
    # Convert data to serializable format (convert numpy arrays or tensors to lists)
    serializable_data = {}
    for key, value in loss_data.items():
        if value is not None:
            if isinstance(value, (list, tuple)):
                serializable_data[key] = [float(v) if hasattr(v, 'item') else v for v in value]
            elif hasattr(value, 'item'):  # For single tensor values
                serializable_data[key] = float(value)
            else:
                serializable_data[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(serializable_data, f, indent=4)
    
    print(f"Loss history data saved to {save_path}")
    return save_path

def plot_loss(loss_history, val_loss_history=None, val_ppl_history=None, save_dir='data/loss_graphs', 
              filename='loss_curve.png', overwrite=False, save_history=False):
    """
    Plot and save the loss curves for training and validation.
    
    Args:
        loss_history (list): History of training loss values
        val_loss_history (list, optional): History of validation loss values
        val_ppl_history (list, optional): History of validation perplexity values
        save_dir (str): Directory to save the plots
        filename (str): Filename for the loss curve plot
        overwrite (bool): Whether to overwrite existing files
        save_history (bool): Whether to save the loss history data to a JSON file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, filename)
    
    if not overwrite and os.path.exists(save_path):
        base, ext = os.path.splitext(filename)
        i = 1
        while os.path.exists(os.path.join(save_dir, f"{base}_{i}{ext}")):
            i += 1
        save_path = os.path.join(save_dir, f"{base}_{i}{ext}")
    
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    
    if val_loss_history is not None:
        plt.plot(val_loss_history, label='Validation Loss')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to {save_path}")
    
    # Plot perplexity if available
    if val_ppl_history is not None:
        ppl_path = os.path.join(save_dir, 'perplexity_curve.png')
        if not overwrite and os.path.exists(ppl_path):
            base, ext = os.path.splitext('perplexity_curve.png')
            i = 1
            while os.path.exists(os.path.join(save_dir, f"{base}_{i}{ext}")):
                i += 1
            ppl_path = os.path.join(save_dir, f"{base}_{i}{ext}")
            
        plt.figure(figsize=(10, 5))
        plt.plot(val_ppl_history, label='Validation Perplexity', color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Perplexity')
        plt.title('Perplexity Curve')
        plt.legend()
        plt.savefig(ppl_path)
        plt.close()
        print(f"Perplexity curve saved to {ppl_path}")
    
    # Save history data if requested
    if save_history:
        loss_data = {
            'train_loss': loss_history,
            'val_loss': val_loss_history,
            'val_ppl': val_ppl_history
        }
        history_filename = os.path.splitext(filename)[0] + '_data.json'
        save_loss_history(loss_data, save_dir, history_filename, overwrite)


