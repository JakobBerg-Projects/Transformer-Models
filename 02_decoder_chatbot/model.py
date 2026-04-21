import torch
import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim),
        )
        self.dropout = nn.Dropout(dropout)
                                  
    def forward(self, x, mask=None):
        # TODO: Implement this method
        normed = self.norm1(x)
        attn_out= self.attn(normed, normed, normed, key_padding_mask=mask)
        x = x + self.dropout(attn_out)

        normed2 = self.norm2(x)
        mlp_out = self.mlp(normed2)
        x = x + self.dropout(mlp_out)

        return x 
        



    

class PositionalEncoding(nn.Module):
    """
    Positional encoding module: adds positional information to the input embeddings.
    """
    def __init__(self, embed_size, max_len):
        super().__init__()
        # TODO: Implement this method
        # Use self.register_bufffer("positional_encoding", positional_encoding) to store the positional encoding (not a parameter)
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        i = torch.arange(0, embed_size, 2).float()
        pe[:, 0::2] = torch.sin(position / (10000 ** (i / embed_size)))
        pe[:, 1::2] = torch.cos(position / (10000 ** (i / embed_size)))

        self.register_buffer('pe', pe.unsqueeze(0)) 

    def forward(self, x):
        # TODO: Implement this method
        # Remember to slice the positional encoding to match the length of the input sequence
        # and to move the positional encoding to the device of the input tensor
        return x + self.pe[:, :x.size(1), :]
        

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_size = config.embed_size
        self.num_layers = config.num_layers 
        self.vocab_size = config.vocab_size
        self.max_len = config.max_len
        self.dropout_p = config.dropout_p
        self.num_heads = config.num_heads
        self.device = config.device

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.pos_encoder = PositionalEncoding(self.embed_size, self.max_len)

        self.layers = nn.ModuleList([DecoderBlock(self.embed_size, self.num_heads, self.dropout_p) for _ in range(self.num_layers)])
        self.fc_out = nn.Linear(self.embed_size, self.vocab_size)

        # Precompute the causal mask and positional encoding
        self.register_buffer("causal_mask", self.generate_causal_mask(self.max_len))

    def forward(self, x, padding_mask=None):
        batch_size, seq_len = x.shape

        # Use the precomputed causal mask (trim to match seq_len)
        attn_mask = self.causal_mask[:seq_len, :seq_len]

        # Embed and add positional encoding
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, attn_mask, padding_mask)

        return self.fc_out(x)

    def generate_causal_mask(self, seq_len):
        """
        Generates an upper triangular mask to prevent attending to future tokens.
        """
        # TODO: Implement this method
        # You can use torch.ones and torch.triu to generate the mask and cast it to a boolean tensor with .bool()
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask


if __name__ == "__main__":
    from tokenizers import Tokenizer
    from torch.nn.functional import cross_entropy

    from config import config
    from utils import get_num_params
    from dataset import QADataset

    model = TransformerModel(config)
    print(f"Number of parameters in the model: {get_num_params(model):,}")

    # Simple forward pass for sanity checking
    tokenizer = Tokenizer.from_file(config.tokenizer_filename)
    dataset = QADataset(config, tokenizer)
    source = dataset[0]["source_sequence"].unsqueeze(0)
    target = dataset[0]["target_sequence"].unsqueeze(0)
    padding_mask = dataset[0]["key_padding_mask"].unsqueeze(0)

    # Forward pass
    out = model(source, padding_mask)
    print("Output shape:", out.shape)
    print("Target shape:", target.shape)
    print("Loss mask shape:", padding_mask.shape)

    # Calculate loss
    loss = cross_entropy(out.transpose(1, 2), target)
    print("Loss:", loss.item())

