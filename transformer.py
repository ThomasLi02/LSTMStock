import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


# Step 1: Load and preprocess the data
data = pd.read_csv("datasets/apple4.csv")  # assuming closing price is in a column named 'Close'
prices = data['prccd'].values
prices = prices.reshape(-1, 1)  # reshaping to have a single feature

# Split data into training and testing
train_data, test_data = train_test_split(prices, test_size=0.2, shuffle=False)

# Step 2: Create DataLoaders
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return self.data.shape[0] - self.seq_length

    def __getitem__(self, idx):
        seq = torch.FloatTensor(self.data[idx:idx+self.seq_length])  # convert to float
        label = torch.FloatTensor(self.data[idx+self.seq_length:idx+self.seq_length+1])  # convert to float
        return seq, label

seq_length = 10
train_dataset = TimeSeriesDataset(train_data, seq_length)
test_dataset = TimeSeriesDataset(test_data, seq_length)

train_loader = DataLoader(train_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Step 3: Define the Transformer Model
class TransformerPredictor(nn.Module):
    def __init__(self, k, num_layers=3):
        super().__init__()
        self.transformer = nn.Transformer(nhead=1, d_model=k, num_encoder_layers=num_layers)
        self.fc = nn.Linear(k, 1)
    
    def forward(self, x):
        out = self.transformer(x, x)  # use x for both src and tgt
        out = self.fc(out[-1])
        return out


# Step 4: Train the Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerPredictor(k=1).to(device)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for seq, labels in train_loader:
        seq = seq.float().to(device)  # convert the seq to float type and move it to the device
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(seq)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")

model.eval()  # set the model to evaluation mode
predicted = []
final = []
with torch.no_grad():
    test_losses = []
    for seq, labels in test_loader:
        seq = seq.transpose(0, 1).to(device)  # transpose seq before feeding it into the model
        labels = labels.to(device)
        output = model(seq)
        predicted.append(output)
        final.append(labels)
        test_loss = criterion(output, labels)
        test_losses.append(test_loss.item())
    avg_test_loss = sum(test_losses) / len(test_losses)
    print(f"Test Loss: {avg_test_loss}")


x = np.arange(0, len(predicted))

# Plot the lines connecting the points
plt.plot(x, predicted, label='Predictions')
plt.plot(x, final, label='Actual')
print(mean_squared_error(predicted, final))

# Add labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Display the plot
plt.show()