import torch
from torch.utils.data import Dataset, DataLoader


class WikiText103Dataset(Dataset):

    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)
        

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        encoded = self.tokenizer.encode(text).ids

        start_token_id = self.tokenizer.token_to_id("[START]")
        pad_token_id = self.tokenizer.token_to_id("[PAD]")
        end_token_id = self.tokenizer.token_to_id("[END]")
        
        # Debug check for None tokens
        if start_token_id is None:
            raise ValueError("'[START]' token not found in tokenizer vocabulary")
        if pad_token_id is None:
            raise ValueError("'[PAD]' token not found in tokenizer vocabulary")
        if end_token_id is None:
            raise ValueError("'[END]' token not found in tokenizer vocabulary")
        
        source = [start_token_id] + encoded[:self.max_length-1] 
        source = source + [pad_token_id] * (self.max_length - len(source)) 

        target = encoded[:self.max_length-1] + [end_token_id]
        target = target + [pad_token_id] * (self.max_length - len(target))  
        
        return (torch.tensor(source), torch.tensor(target))


# data_subset = dataset['train'].select(range(1000)) 
# train_dataset = WikiText103Dataset(data_subset, tokenizer, max_length=seq_len)
# dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)