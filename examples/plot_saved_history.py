import os
import sys
import argparse
from pathlib import Path

# Add project root to path
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.append(os.getcwd())

from performer.utils import load_loss_history, plot_loaded_losses, compare_runs

def main():
    parser = argparse.ArgumentParser(description='Plot saved loss histories')
    parser.add_argument('--file', type=str, help='Path to the loss history JSON file')
    parser.add_argument('--compare', nargs='+', help='List of JSON files to compare')
    parser.add_argument('--labels', nargs='+', help='Labels for the compared runs')
    parser.add_argument('--metrics', nargs='+', 
                        choices=['train_loss', 'val_loss', 'val_ppl', 'lr'],
                        help='Metrics to plot in comparison mode')
    parser.add_argument('--save_dir', type=str, default='data/plots',
                        help='Directory to save the plots')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not display plots (only save them)')
    
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Handle comparison mode
    if args.compare:
        if args.labels and len(args.labels) != len(args.compare):
            print("Error: Number of labels must match the number of files to compare")
            return
        
        print(f"Comparing loss histories from {len(args.compare)} runs...")
        compare_runs(
            args.compare, 
            labels=args.labels, 
            metrics=args.metrics,
            save_dir=args.save_dir,
            show=not args.no_show
        )
        return
    
    # Single file mode
    if not args.file:
        # If no file specified, try to find the latest one
        data_dir = Path('./data/training_results')
        if data_dir.exists():
            json_files = list(data_dir.glob('*.json'))
            if json_files:
                latest_file = max(json_files, key=os.path.getmtime)
                args.file = str(latest_file)
                print(f"No file specified. Using latest: {args.file}")
            else:
                print("Error: No JSON files found in data/training_results/")
                return
        else:
            print("Error: No file specified and data/training_results/ directory not found")
            return
    
    # Load and plot the single file
    try:
        print(f"Loading loss history from {args.file}...")
        data = load_loss_history(args.file)
        
        # Print some statistics
        if 'train_loss' in data:
            print(f"Training epochs: {len(data['train_loss'])}")
            print(f"Final training loss: {data['train_loss'][-1]:.4f}")
        
        if 'val_loss' in data and data['val_loss']:
            print(f"Final validation loss: {data['val_loss'][-1]:.4f}")
        
        if 'val_ppl' in data and data['val_ppl']:
            print(f"Final validation perplexity: {data['val_ppl'][-1]:.2f}")
            
        # Plot the loaded data
        plot_loaded_losses(data, save_dir=args.save_dir, show=not args.no_show)
        
    except Exception as e:
        print(f"Error loading or plotting loss history: {e}")
        return

if __name__ == "__main__":
    main() 