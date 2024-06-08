import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, nhead=3, dim_feedforward=256, dropout=0.1):
        super(Autoencoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Input has shape (batch_size, input_dim)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = torch.from_numpy(x).float()  # Convert NumPy array to PyTorch tensor
        encoded = self.encoder(x.unsqueeze(1))  # Add a dummy sequence dimension
        encoded = encoded.squeeze(1)  # Remove the dummy sequence dimension
        hidden = self.decoder(encoded)
        output = self.output_layer(hidden)
        return output

def train_autoencoder(model, train_data, num_epochs, batch_size, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i in range(0, len(train_data), batch_size):
            inputs = train_data[i:i+batch_size]
            # inputs = torch.stack(inputs)

            optimizer.zero_grad()
            outputs = model(inputs)
            inputs=torch.from_numpy(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
    with open("TA_losses.txt", "a") as file:
        loss_value = running_loss / (len(train_data) / batch_size)
        formatted_loss = "{:,.2f}".format(loss_value)
        file.write(formatted_loss+",")

    print(f"Epoch {epoch+1} loss: {running_loss / (len(train_data) / batch_size)}")