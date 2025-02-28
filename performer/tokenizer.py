from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def create_tokenizer(vocab_size):
    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[START]", "[END]"]
    
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )
    
    for token in special_tokens:
        tokenizer.add_special_tokens([token])
    
    return tokenizer, trainer