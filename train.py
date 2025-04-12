import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train(model, dataloader, num_epochs=10, lr=1e-4, device="cuda"):
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            wav = batch.to(device)  # [B, 1, T]
            optimizer.zero_grad()

            out = model(wav)  # [B, 1, T]
            loss = mse_loss(out, wav)  
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.2f}")
        
