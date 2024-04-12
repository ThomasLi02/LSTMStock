import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load the dataset (example)
data = pd.read_csv('datasets/apple4.csv')  # Replace 'apple_stock_prices.csv' with your actual dataset file

# Create a shifted 'Label' column for next day's closing price
data['Label'] = data['prccd'].shift(-1)

# Drop the last row as it has a NaN 'Label'
data = data.dropna()

# Split the dataset into input features (X) and labels (y)
X = data[['prccd']].values  # Use closing prices as input features
y = data['Label'].values  # Use next day's closing prices as labels

# Normalize the input features and labels

# Split the dataset into training and testing sets

X_min, X_max = X.min(), X.max()
y_min, y_max = y.min(), y.max()
X = (X - X_min) / (X_max - X_min)
y = (y - y_min) / (y_max - y_min)


X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=False, test_size=0.2, random_state=42)


print(X_test.flatten())
print(y_test)
# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader for training and testing sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=32, nhead=2, num_layers=10):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        src = src.unsqueeze(0)  # Add a dimension
        tgt = tgt.unsqueeze(0)  # Add a dimension
        output = self.transformer(src, tgt)
        output = output.mean(dim=0)  # Aggregate transformer outputs
        output = self.fc(output)
        return output

# Create the Transformer model
input_size = 1  # Number of input features
model = TransformerModel(input_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training loop
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, inputs)  # Use the same input tensor as the target
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Evaluation
model.eval()
total_loss = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        print(inputs)
        outputs = model(inputs, inputs)  # Use the same input tensor as the target
        loss = criterion(outputs, labels)
        outputs = outputs.flatten()
        total_loss += loss.item()
        outputs = outputs.cpu().numpy() * (y_max - y_min) + y_min
        labels = labels.cpu().numpy() * (y_max - y_min) + y_min
        print("Predicted prices:", outputs)
        print("Actual prices:", labels)

mean_loss = total_loss / len(test_loader)
print(f"Test Mean Squared Error: {mean_loss}")
