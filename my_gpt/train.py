import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import hashlib

class GPTDatasetV1(Dataset):
    def __init__(self, input_ids, target_ids):
        self.input_ids = input_ids
        self.target_ids = target_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def get_cache_path(txt, max_length, stride):
    # Create a unique hash for the input parameters
    params = f"{txt[:50]}_{txt[-50:]}_{max_length}_{stride}".encode('utf-8')
    hash_value = hashlib.md5(params).hexdigest()
    return f"tokenized_{hash_value}.pkl"


def create_dataloader(txt, tokenizer, batch_size=4, max_length=256, 
                      stride=128, shuffle=True, drop_last=True, 
                      num_workers=0, cache_dir="cache"):
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, get_cache_path(txt, max_length, stride))
    
    # Try to load from cache first
    if os.path.exists(cache_path):
        print(f"Loading tokenized data from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            token_ids = pickle.load(f)
    else:
        # Tokenize the entire text
        print("Tokenizing text...")
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        # Save to cache
        print(f"Saving tokenized data to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(token_ids, f)
        
    print("Creating dataset...")
    input_ids = []
    target_ids = []

    # Use a sliding window to chunk the book into overlapping sequences of max_length
    for i in range(0, len(token_ids) - max_length, stride):
        input_chunk = token_ids[i:i + max_length]
        target_chunk = token_ids[i + 1: i + max_length + 1]
        input_ids.append(torch.tensor(input_chunk))
        target_ids.append(torch.tensor(target_chunk))

    # Create dataset
    dataset = GPTDatasetV1(input_ids, target_ids)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader