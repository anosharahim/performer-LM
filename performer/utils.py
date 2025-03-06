import json
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np

def load_loss_history(filepath):
    """
    Load loss history data from a JSON file.
    
    Args:
        filepath (str): Path to the JSON file containing loss history data
        
    Returns:
        dict: Dictionary containing the loss history data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Loss history file not found at {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data

def plot_loaded_losses(data, save_dir=None, show=True):
    """
    Plot loss histories from loaded data.
    
    Args:
        data (dict): Dictionary containing loss history data
        save_dir (str, optional): Directory to save the plots. If None, plots are not saved.
        show (bool): Whether to display the plots interactively
        
    Returns:
        tuple: Paths to the saved plots (if save_dir is provided)
    """
    saved_paths = []
    
    # Create save directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Plot training and validation loss
    if 'train_loss' in data:
        plt.figure(figsize=(10, 5))
        plt.plot(data['train_loss'], label='Training Loss')
        
        if 'val_loss' in data and data['val_loss'] is not None:
            plt.plot(data['val_loss'], label='Validation Loss')
            
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        
        if save_dir:
            save_path = os.path.join(save_dir, 'loss_curve_replot.png')
            plt.savefig(save_path)
            saved_paths.append(save_path)
            print(f"Loss curve saved to {save_path}")
            
        if show:
            plt.show()
        else:
            plt.close()
    
    # Plot validation perplexity
    if 'val_ppl' in data and data['val_ppl'] is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(data['val_ppl'], label='Validation Perplexity', color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Perplexity')
        plt.title('Perplexity Curve')
        plt.legend()
        
        if save_dir:
            save_path = os.path.join(save_dir, 'perplexity_curve_replot.png')
            plt.savefig(save_path)
            saved_paths.append(save_path)
            print(f"Perplexity curve saved to {save_path}")
            
        if show:
            plt.show()
        else:
            plt.close()
    
    # Plot learning rate if available
    if 'lr' in data and data['lr'] is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(data['lr'], label='Learning Rate', color='purple')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        
        if save_dir:
            save_path = os.path.join(save_dir, 'lr_curve_replot.png')
            plt.savefig(save_path)
            saved_paths.append(save_path)
            print(f"Learning rate curve saved to {save_path}")
            
        if show:
            plt.show()
        else:
            plt.close()
    
    return saved_paths if save_dir else None

def compare_runs(filepaths, labels=None, metrics=None, save_dir=None, show=True):
    """
    Compare loss histories from multiple training runs.
    
    Args:
        filepaths (list): List of paths to JSON files containing loss history data
        labels (list, optional): List of labels for each run
        metrics (list, optional): List of metrics to plot ('train_loss', 'val_loss', 'val_ppl', 'lr').
                                  If None, all available metrics are plotted.
        save_dir (str, optional): Directory to save the plots. If None, plots are not saved.
        show (bool): Whether to display the plots interactively
        
    Returns:
        tuple: Paths to the saved plots (if save_dir is provided)
    """
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(filepaths))]
    
    if len(labels) != len(filepaths):
        raise ValueError("Number of labels must match number of filepaths")
    
    # Load all run data
    all_data = []
    for filepath in filepaths:
        all_data.append(load_loss_history(filepath))
    
    # Determine available metrics if not specified
    if metrics is None:
        metrics = set()
        for data in all_data:
            metrics.update([k for k in data.keys() if isinstance(data[k], list)])
        metrics = list(metrics)
    
    saved_paths = []
    
    # Create save directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Plot each metric
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        for i, (data, label) in enumerate(zip(all_data, labels)):
            if metric in data and data[metric] is not None:
                plt.plot(data[metric], label=f"{label}")
        
        plt.xlabel('Epochs')
        metric_name = ' '.join(word.capitalize() for word in metric.split('_'))
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} Comparison')
        plt.legend()
        
        if save_dir:
            save_path = os.path.join(save_dir, f'{metric}_comparison.png')
            plt.savefig(save_path)
            saved_paths.append(save_path)
            print(f"{metric_name} comparison saved to {save_path}")
            
        if show:
            plt.show()
        else:
            plt.close()
    
    return saved_paths if save_dir else None 