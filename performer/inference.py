import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os
import json
from pathlib import Path

from performer.model import Transformer
from performer.tokenizer import create_tokenizer


def load_model(model_path, config):
    """
    Load a saved model from the given path
    """
    # Create a new model with the same configuration
    model = Transformer(
        embedding_dim=config['embedding_dim'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        dropout_rate=config['dropout_rate'],
        num_layers=config['num_layers'],
        vocab_size=config['vocab_size']
    )
    
    # Load the trained weights with weights_only=True for safety
    try:
        model.load_state_dict(torch.load(model_path, map_location=config['device'], weights_only=True))
        print("Model loaded successfully with exact parameter match")
    except Exception as e:
        print(f"Warning: Exact model loading failed: {e}")
        print("Attempting to load with strict=False to handle parameter mismatches...")
        
        # Try loading with strict=False to handle partial parameter matches
        try:
            model_state = torch.load(model_path, map_location=config['device'], weights_only=True)
            model.load_state_dict(model_state, strict=False)
            print("Model loaded with partial parameter match (some parameters may be missing or mismatched)")
        except Exception as e:
            raise ValueError(f"Failed to load model even with non-strict loading: {e}\n"
                            "Make sure your model configuration matches the saved model's configuration.")
    
    model.eval()  # Set model to evaluation mode
    model.to(config['device'])
    
    return model


def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, top_k=50, top_p=0.9, device='cpu'):
    """
    Generate text from a prompt using the trained model
    
    Args:
        model: The trained transformer model
        tokenizer: The tokenizer used during training
        prompt: Text prompt to start generation
        max_length: Maximum number of tokens to generate
        temperature: Controls randomness (lower = more deterministic)
        top_k: Number of highest probability tokens to keep for sampling
        top_p: Cumulative probability cutoff for nucleus sampling
        device: Device to run inference on (cuda or cpu)
    
    Returns:
        The generated text as a string
    """
    model.eval()
    
    # Encode the prompt
    start_token_id = tokenizer.token_to_id("[START]")
    encoded_prompt = tokenizer.encode(prompt).ids
    
    # Add start token if needed
    if start_token_id not in encoded_prompt:
        input_ids = [start_token_id] + encoded_prompt
    else:
        input_ids = encoded_prompt
    
    # Convert to tensor
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    
    # Start with just the prompt for the decoder input
    # Since this is a transformer model where the decoder needs to be primed
    # with a starting token
    decoder_input = torch.tensor([[start_token_id]]).to(device)
    
    # Generate text token by token
    with torch.no_grad():
        for _ in tqdm(range(max_length), desc="Generating text"):
            # Forward pass
            outputs = model(input_ids, decoder_input)
            
            # Get the predictions for the last token only
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = torch.topk(next_token_logits, k=top_k)[0][:, -1].unsqueeze(-1)
                next_token_logits[next_token_logits < indices_to_remove] = float('-inf')
            
            # Apply top-p/nucleus filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[:, indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check for end token
            end_token_id = tokenizer.token_to_id("[END]")
            if next_token.item() == end_token_id:
                break
                
            # Add the predicted token to the decoder input for the next iteration
            decoder_input = torch.cat((decoder_input, next_token), dim=1)
    
    # Decode the generated tokens, excluding start and end tokens
    generated_tokens = decoder_input.squeeze().tolist()[1:]  # Remove the start token
    generated_text = tokenizer.decode(generated_tokens)
    
    return prompt + generated_text


def save_model(model, save_path):
    """
    Save the model to the given path
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained transformer model")
    parser.add_argument("--model_path", type=str, default="data/trained_model.pt", 
                        help="Path to saved model")
    parser.add_argument("--config_path", type=str, default="data/model_config.json",
                        help="Path to model configuration file")
    parser.add_argument("--tokenizer_path", type=str, default="data/tokenizer.json", 
                        help="Path to tokenizer")
    parser.add_argument("--prompt", type=str, default="The transformer model", 
                        help="Text prompt to start generation")
    parser.add_argument("--max_length", type=int, default=50,
                        help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for sampling (lower is more deterministic)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Number of highest probability tokens to keep")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Cumulative probability cutoff for nucleus sampling")
    
    # Model configuration parameters (default values, will be overridden by config file if it exists)
    parser.add_argument("--embedding_dim", type=int, default=64,
                        help="Model embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--ff_dim", type=int, default=256,
                        help="Feed-forward dimension")
    parser.add_argument("--dropout_rate", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of transformer layers")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check paths
    model_path = Path(args.model_path)
    tokenizer_path = Path(args.tokenizer_path)
    config_path = Path(args.config_path)
    
    # Load tokenizer
    if not tokenizer_path.exists():
        print(f"Tokenizer not found at {tokenizer_path}. Please check the path.")
        return
    
    tokenizer, _ = create_tokenizer(0, load_path=str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    print(f"Loaded tokenizer with vocabulary size: {vocab_size}")
    
    # Check if model exists
    if not model_path.exists():
        print(f"Model not found at {model_path}. Please train the model first or provide the correct path.")
        return
    
    # Try to load model configuration from file if available
    if config_path.exists():
        try:
            print(f"Loading model configuration from {config_path}")
            with open(config_path, 'r') as f:
                model_config = json.load(f)
            
            # Add vocab_size and device to config
            model_config["vocab_size"] = vocab_size
            model_config["device"] = device
            
            print(f"Using config: embedding_dim={model_config['embedding_dim']}, " +
                  f"num_layers={model_config['num_layers']}, " +
                  f"num_heads={model_config['num_heads']}, " +
                  f"ff_dim={model_config['ff_dim']}")
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Using command line parameters instead.")
            model_config = {
                "embedding_dim": args.embedding_dim,
                "num_heads": args.num_heads,
                "ff_dim": args.ff_dim,
                "dropout_rate": args.dropout_rate,
                "num_layers": args.num_layers,
                "vocab_size": vocab_size,
                "device": device
            }
    else:
        print("No configuration file found. Using command line parameters.")
        model_config = {
            "embedding_dim": args.embedding_dim,
            "num_heads": args.num_heads,
            "ff_dim": args.ff_dim,
            "dropout_rate": args.dropout_rate,
            "num_layers": args.num_layers,
            "vocab_size": vocab_size,
            "device": device
        }
    
    # Load model
    try:
        model = load_model(str(model_path), model_config)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Generate text from prompt
    generated_text = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device
    )
    
    print("\nGenerated text:")
    print(f"{generated_text}")


if __name__ == "__main__":
    main() 