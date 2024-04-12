import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam
import tensorflow as tf
import random

import tensorflow as tf


random.seed(123)
np.random.seed(123)

# Set the random seed for TensorFlow/Keras
tf.random.set_seed(123)

# Load the dataset
data = pd.read_csv('datasets/tech_sector.csv')

tic_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN','ORCL', 'PYPL', 'NVDA', 'TSM', 'V']
tic_lengths = []

for tic in tic_list:
    filtered_data = data[data['tic'] == tic]
    tic_lengths.append(len(filtered_data))

print(tic_lengths)
data = data[data['tic'].isin(tic_list)]

data = np.array(data)
print(len(data))

binary = []
for i in range(1008):
    sums = 0
    for j in range(9):
        sums += float(data[i+1008*j][4])
    binary.append(sums)


binary = np.array(binary)
print(len(binary))


# Calculate the binary target variable
binary = np.array(binary)

# Preprocess the data
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(binary.reshape(-1, 1))

# Split the data into train and test sets
train_size = int(len(scaled_data) * 7/ 8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]


# Create sliding window data
def create_windowed_data(data, window_size):
    x_data = []
    y_data = []
    for i in range(window_size, len(data)):
        x_data.append(data[i-window_size:i, 0])
        y_data.append(data[i, 0])
    return np.array(x_data), np.array(y_data)

window_size = 10
x_train, y_train = create_windowed_data(train_data, window_size)
x_test, y_test = create_windowed_data(test_data, window_size)

# Reshape the input to be 3D (samples, time steps, features)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=256, return_sequences=False, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.3))
model.add(Dense(units=1))

# model = Sequential()
# model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# model.add(LSTM(units=100,return_sequences=True))
# model.add(LSTM(units=100))
# model.add(Dense(units=1))

# Compile and train the model
optimizer = Adam(learning_rate=0.004)
model.compile(optimizer=optimizer, loss='mean_squared_error')
history = model.fit(x_train, y_train, epochs=1000, batch_size=32)

# Evaluate the model
predictions = model.predict(x_test)
predictions_inverse = np.copy(predictions)
column_index = 0  # Index of the column to inverse transform

predictions_inverse[:, column_index] = scaler.inverse_transform(predictions[:, column_index].reshape(-1, 1)).flatten()

# Print the inverse transformed predictions
print(predictions_inverse[:, column_index])
predictions = predictions_inverse

y_test = scaler.inverse_transform(y_test.reshape(len(y_test),1))
count = 0 
for i in range(1,len(y_test)):
    print(predictions[i],y_test[i])
    if np.sign(y_test[i][0]-y_test[i-1][0]) == np.sign(x_test[i][0]-x_test[i-1][0]):
        count += 1
print(count/(len(y_test)-1))


import matplotlib.pyplot as plt
# print(len(predictions) == len(y_test))

# Generate x-values based on the length of the arrays
# Generate x-values for predictions and actual values separately
x_predictions = np.arange(window_size, window_size + len(predictions))
x_actual = np.arange(window_size, window_size + len(y_test))

# Plot the lines connecting the points
plt.plot(x_predictions, predictions, label='Predictions')
plt.plot(x_actual, y_test, label='Actual')

# Add labels and legend
plt.title('LSTM for Tech Companies')
plt.xlabel('Days')
plt.ylabel('Total Price of 10 Tech')
plt.legend()

# Display the plot


print(mean_squared_error(predictions, y_test))
# Display the plot
plt.show()          


loss = history.history['loss']
epochs = range(1, 1000 + 1)

plt.plot(epochs, loss, 'b', label='Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



