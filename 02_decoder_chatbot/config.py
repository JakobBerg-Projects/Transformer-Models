import torch
from types import SimpleNamespace

"""
This file contains the project configuration. To access the config from other scripts, simply import it by `from config import config` and then access the config values using `config.<attribute>`. For example, to access the batch size, use `config.batch_size`.
"""

config = SimpleNamespace(**{
    # General config
    "seed": 0,
    "dataset": "gooaq_subset",
    "split": "train",
    "device": "cpu" if not torch.cuda.is_available() else "cuda",
    
    # Tokenizer config
    "vocab_size": 20_000,
    "min_frequency": 5,
    "unk_token": "[UNK]",
    "sep_token": "[SEP]",
    "end_token": "[END]",
    "pad_token": "[PAD]",
    "tokenizer_filename": "temp/tokenizer.json",
    
    # Model config
    "embed_size": 512,
    "num_heads": 8,
    "num_layers": 5,
    "dropout_p": 0.1,

    # Training config
    "max_len": 128,
    "model_train_fraction": 1.0,
    "batch_size": 128,
    "dataloader_num_workers": 2,
    "lr": 1e-4,
    "num_epochs": 5,
    "model_filename": "temp/model.pth",
    "optimizer_filename": "temp/optimizer.pth",
    })


# Uncomment the below code to use a tiny model for testing your code before GPU training
"""
if config.device == "cpu":
    config.vocab_size = 5000
    config.embed_size = 16 
    config.num_heads = 2
    config.num_layers = 2
    config.batch_size = 32
    config.num_epochs = 1
    config.max_len = 32
    config.model_train_fraction = 0.1
"""
