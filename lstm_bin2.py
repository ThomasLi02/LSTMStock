import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam

# Load the dataset
data = pd.read_csv('datasets/apple4.csv')
data = data[data['tic'] == 'AAPL']

# Shift the 'prccd' column by one to align the current day's closing price with the previous day's
data['prccd_shifted'] = data['prccd'].shift(1)

binary = np.array(data['prccd']) - np.array(data['prccd_shifted'])
binary = np.sign(binary)
print(binary)
binary = np.nan_to_num(binary, nan=0)  # Replace potential NaNs resulting from the shift operation

# Loop over the binary values and replace anything that is not 1 with 0
binary = [1 if b == 1 else 0 for b in binary]

binary = np.array(binary)

scaled_data = binary.reshape(-1,1)
train_size = int(len(scaled_data) * 7 / 8)
train_data = scaled_data[1:train_size]
test_data = scaled_data[train_size:]

def create_windowed_data(data, window_size):
    x_data = []
    y_data = []
    for i in range(window_size, len(data)):
        x_data.append(data[i-window_size:i, 0])
        y_data.append(data[i, 0])
    return np.array(x_data), np.array(y_data)


window_size = 3
x_train, y_train = create_windowed_data(train_data, window_size)
x_test, y_test = create_windowed_data(test_data, window_size)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))



model = Sequential()
model.add(LSTM(units=16, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.3))
model.add(LSTM(units=16, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=16, return_sequences=True))
model.add(Dropout(0.3))
model.add(Dense(units=1, activation='sigmoid'))  # Add activation='sigmoid' for binary classification

optimizer = Adam(learning_rate=0.005)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=100, batch_size=32)



predictions = model.predict(x_test)
predictions = predictions.reshape(len(predictions),1)
print(predictions)
threshold = 0.5
binary_predictions = np.array([1 if p > threshold else 0 for p in predictions])
binary_actuals = np.array([1 if p > threshold else 0 for p in y_test])

accuracy = np.mean(binary_predictions == binary_actuals) * 100

print("Accuracy: %.2f%%" % accuracy)


print(f"Accuracy: {accuracy:.4f}")

import matplotlib.pyplot as plt


loss = history.history['loss']
epochs = range(1, 100 + 1)

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.title('LSTM Binary Loss per Epoch for Apple')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()                          