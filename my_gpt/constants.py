GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 256,  # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False,      # Query-Key-Value bias
    "flash": True,          # Use Flash Attention or not
}

TRAIN_SETTINGS = {
    "learning_rate": 3e-4,
    "num_epochs": 1,
    "batch_size": 96,
    "weight_decay": 0.1,
    "train_test_ratio": 0.9,
}