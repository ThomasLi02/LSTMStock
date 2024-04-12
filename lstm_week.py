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
data = pd.read_csv('apple4.csv')
data = data[data['tic'] == 'AAPL']
lists = [ 'prccd','prcod','prchd', 'prcld' ]
print(data[lists])

# data = data[1086:2160]

# for i in range(1089):
#     sums = 0
#     for j in range(41):
#         sums += float(data[i+1089*j][13]) - float(data[i+1089*j][16])

#     binary.append(sums)
# binary = np.array(binary)


# Calculate the binary target variable

binary = np.array(data['prccd'])

# Preprocess the data
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(binary.reshape(-1, 1))

# Split the data into train and test sets
train_size = int(len(scaled_data) * 3/ 4)
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
model.fit(x_train, y_train, epochs=200, batch_size=32)

# Evaluate the model
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

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
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Display the plot


print(mean_squared_error(predictions, y_test))
# Display the plot
plt.show()          
