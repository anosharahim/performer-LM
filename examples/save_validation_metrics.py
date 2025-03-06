import json
import sys
from pathlib import Path

# Path to save validation metrics
data_dir = Path("./data")
val_metrics_path = data_dir / "val_metrics.json"

# Get the validation perplexity from the user
if len(sys.argv) > 1:
    try:
        val_ppl = float(sys.argv[1])
    except ValueError:
        print("Error: Please provide a valid number for validation perplexity")
        sys.exit(1)
else:
    val_ppl_str = input("Enter the final validation perplexity from your training run: ")
    try:
        val_ppl = float(val_ppl_str)
    except ValueError:
        print("Error: Please enter a valid number")
        sys.exit(1)
        
# Optional validation loss
val_loss = None
val_loss_str = input("Enter the final validation loss (optional, press Enter to skip): ")
if val_loss_str.strip():
    try:
        val_loss = float(val_loss_str)
    except ValueError:
        print("Warning: Invalid validation loss. Continuing without it.")

# Create the metrics dictionary
val_metrics = {
    "final_val_perplexity": val_ppl,
}

if val_loss is not None:
    val_metrics["final_val_loss"] = val_loss

# Save to file
with open(val_metrics_path, 'w') as f:
    json.dump(val_metrics, f, indent=4)
    
print(f"Validation metrics saved to {val_metrics_path}")
print("You can now run test_evaluation.py to compare test results with these validation metrics.") 