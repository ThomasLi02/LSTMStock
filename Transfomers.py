import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import random

import tensorflow as tf


random.seed(123)
np.random.seed(123)

# Set the random seed for TensorFlow/Keras
tf.random.set_seed(123)
torch.manual_seed(123)
# Step 1: Preprocess the Data
stock_data = pd.read_csv("datasets/apple4.csv")

features = ["prccd", "prchd", "prcld", "prcod"]
target = "prccd"
final = stock_data["prccd"]

x = int(len(final) * 7/ 8)
final = final[x:]
final = final[10:]


#
scaler = MinMaxScaler(feature_range=(0,1))
stock_data[features] = scaler.fit_transform(stock_data[features])

train_data, test_data = stock_data[:x], stock_data[x:]

# Step 2: Prepare the Data for Transformers Model
class StockDataset(Dataset):
    def __init__(self, data, seq_length, target):
        self.data = data
        self.seq_length = seq_length
        self.target = target

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        seq_data = self.data.iloc[index : index + self.seq_length][features].values
        target = self.data.iloc[index + self.seq_length][self.target]
        return torch.tensor(seq_data, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

seq_length = 10

train_dataset = StockDataset(train_data, seq_length, target)
test_dataset = StockDataset(test_data, seq_length, target)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Step 3: Define and Train the Transformers Model
class StockTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(StockTransformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=input_size,
            nhead=4,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size,
        )
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.transformer(x, x)
        x = self.fc(x[-1])
        return x

input_size = len(features)
hidden_size = 128
num_layers = 3
lr = 0.0003
num_epochs = 200

model = StockTransformer(input_size, hidden_size, num_layers)

criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=lr)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


train_losses = []
test_losses = []

for epoch in range(num_epochs):
    print('Epoch: ', epoch)

    model.train()
    total_loss = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs.transpose(0, 1))
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(total_loss)

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()

predicted = []
see = []
print(len(test_loader))
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs.transpose(0, 1))
        column1 = np.random.randn(len(outputs), 1)  # Example column of shape (64, 1)
        column2 = np.random.randn(len(outputs), 1)  # Example column of shape (64, 1)
        column3 = np.random.randn(len(outputs), 1)  # Example column of shape (64, 1)
        see += list(outputs.reshape(len(outputs),))
        # Stack the columns horizontally
        new_array = np.hstack((outputs, column1, column2, column3))
        predicted_values=new_array
        # Denormalize the predicted values
        predicted_values = scaler.inverse_transform(new_array.squeeze())
        predicted_values = predicted_values[:,0]
        targets = targets.reshape(len(targets),1)
        new_array = np.hstack((targets, column1, column2, column3))
        # Denormalize the actual target values
        actual_values = scaler.inverse_transform(new_array.squeeze())
        actual_values=new_array
        actual_values = actual_values[:, 0]
        predicted += list(predicted_values.reshape(len(actual_values),))

        # Reshape the denormalized values to match the original shape

        # Print the predicted and actual values

print(see)

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
# Step 4: Plot the Loss Curves