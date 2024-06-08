import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import gc

import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns


torch.cuda.empty_cache()
gc.collect()




# last_layer = torch.load('./clean_params/params_1.pt')
# print(last_layer.weight)

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, num_heads=1, dropout=0.1):
        super(TransformerAutoencoder, self).__init__()
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_layers)
        self.decoder_layers = nn.TransformerDecoderLayer(d_model=input_size, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(self.decoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        encoder_output = self.encoder(input)
        
        decoder_output = self.decoder(torch.zeros_like(input), encoder_output)
        
        output = self.fc(decoder_output)
        return output


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_weights = nn.Linear(hidden_size, 1)

    def forward(self, encoder_outputs):
        attn_scores = self.attention_weights(encoder_outputs)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context_vector = torch.sum(encoder_outputs * attn_weights, dim=1)
        return context_vector, attn_weights

class RNNAttentionAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNAttentionAutoencoder, self).__init__()
        self.hidden_size = hidden_size
        self.encoder_rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.decoder_rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        encoder_output, _ = self.encoder_rnn(input)
        
        context_vector, _ = self.attention(encoder_output)
        
        decoder_input = context_vector.unsqueeze(1).repeat(1, input.size(1), 1)
        decoder_output, _ = self.decoder_rnn(decoder_input)
        
        output = self.fc(decoder_output)
        return output


class CustomDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = os.listdir(folder_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.folder_path, file_name)
        tensor_data = torch.load(file_path)  # Load tensor from file
        return tensor_data.weight

folder_path = 'clean_params'

custom_dataset = CustomDataset(folder_path)

batch_size = 64
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Example usage:
# print(len(data_loader))

input_size = 6144
hidden_size = 1024
output_size = 6144

# autoencoder = TransformerAutoencoder(input_size, hidden_size, output_size)

autoencoder = RNNAttentionAutoencoder(input_size, hidden_size, output_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = autoencoder.to(device)
criterion = nn.MSELoss()


# training 
"""

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs in dataloader:  # Assuming your DataLoader yields (input, target) pairs
        optimizer.zero_grad()

        # Forward pass
        outputs = autoencoder(inputs)

        # Compute loss
        loss = criterion(outputs, inputs)  # Reconstruction loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader)}")

print("Training finished!")


save_path = "trained_autoencoder.pth"
torch.save(autoencoder.state_dict(), save_path)
print("Model saved successfully.")

"""
load_path = "trained_autoencoder.pth"
autoencoder = RNNAttentionAutoencoder(input_size, hidden_size, output_size)
autoencoder.load_state_dict(torch.load(load_path))
autoencoder=autoencoder.to(device)

print("Model loaded successfully.")


# last_layer = torch.load('./clean_params/params_1.pt')
# print(last_layer.weight)

reconstruction_errors = []

folder_path = './clean_params'
for file_name in os.listdir(folder_path):
    malicious_tensor = torch.load(os.path.join(folder_path, file_name)).to(device).weight
    
    with torch.no_grad():
        autoencoder.eval() 
        output = autoencoder(malicious_tensor.unsqueeze(0))

    reconstruction_error = criterion(output.squeeze(0), malicious_tensor).item()
    reconstruction_errors.append(reconstruction_error)

print("Mean - ", torch.tensor(reconstruction_errors).mean().item())
print("SD - ", torch.tensor(reconstruction_errors).std().item())
f_reconstruction_errors = [f"{error:.6f}" for error in reconstruction_errors]
print(", ".join(f_reconstruction_errors))
# cumlutative distribution func

norm_cdf = scipy.stats.norm.cdf(reconstruction_errors) # calculate the cdf - also discrete

# plot the cdf
sns.lineplot(x=reconstruction_errors, y=norm_cdf)
plt.show()
