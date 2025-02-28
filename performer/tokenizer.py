from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path

def create_tokenizer(vocab_size, load_path=None):
    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[START]", "[END]"]
    
    if load_path and Path(load_path).exists():
        tokenizer = Tokenizer.from_file(str(load_path))
        if hasattr(tokenizer, 'model') and hasattr(tokenizer.model, 'unk_token'):
            tokenizer.model.unk_token = "[UNK]"
        return tokenizer, None
    
    # create new tokenizer if none exists
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )
    
    tokenizer.add_special_tokens(special_tokens)
    
    return tokenizer, trainer