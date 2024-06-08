import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Your model architecture
class TransformerAutoEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, nhead, num_encoder_layers, num_decoder_layers):
        super(TransformerAutoEncoder, self).__init__()

        # Original input dimension should be the flattened size
        self.embedding = nn.Linear(input_dim , embed_dim)  # Assuming that the second dimension is 10 here
        self.pos_encoder = PositionalEncoding(embed_dim)

        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(d_model=embed_dim, nhead=nhead)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.output_layer = nn.Linear(embed_dim, input_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.embedding(x)  # Linear embedding
        x = self.pos_encoder(x)  # Positional encoding
        
        memory = self.transformer_encoder(x)  # Encoding
        output = self.transformer_decoder(x, memory)  # Decoding
        
        output = self.output_layer(output)  # Output layer
        
        return output
# Example usage
print('\a')




# input_size = (6144, 10)  # input_size[1]*10 would be the size after flattening
# embed_dim = 256
# nhead = 8
# num_encoder_layers = 3
# num_decoder_layers = 3

# model = TransformerAutoEncoder(input_size[1]*10, embed_dim, nhead, num_encoder_layers, num_decoder_layers)
# input_tensor = torch.randn(input_size)  # Example input
# output = model(input_tensor)
# print(output.shape)