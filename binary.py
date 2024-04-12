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


prob3 = {
    (0, 0, 0): '0/0',
    (0, 0, 1): '0/0',
    (0, 1, 0): '0/0',
    (0, 1, 1): '0/0',
    (1, 0, 0): '0/0',
    (1, 0, 1): '0/0',
    (1, 1, 0): '0/0',
    (1, 1, 1): '0/0'
}
prob2 = {
    (0, 0): '0/0',
    (0, 1): '0/0',
    (1, 0): '0/0',
    (1, 1): '0/0'
}
prob4 = {
    (0, 0, 0, 0): '0/0',
    (0, 0, 0, 1): '0/0',
    (0, 0, 1, 0): '0/0',
    (0, 0, 1, 1): '0/0',
    (0, 1, 0, 0): '0/0',
    (0, 1, 0, 1): '0/0',
    (0, 1, 1, 0): '0/0',
    (0, 1, 1, 1): '0/0',
    (1, 0, 0, 0): '0/0',
    (1, 0, 0, 1): '0/0',
    (1, 0, 1, 0): '0/0',
    (1, 0, 1, 1): '0/0',
    (1, 1, 0, 0): '0/0',
    (1, 1, 0, 1): '0/0',
    (1, 1, 1, 0): '0/0',
    (1, 1, 1, 1): '0/0'
}

dicts = [prob2,prob3,prob4]



def getFraction(length, window_size, train_date, test_data, lists):
    x_train, y_train = create_windowed_data(train_data, window_size)
    x_test, y_test = create_windowed_data(test_data, window_size)
    prob = lists[length-2]


    for i in range(len(x_train)):
        key = tuple(x_train[i])
        theIndex = prob[key].index('/')
        num = prob[key][:theIndex]
        den = prob[key][theIndex+1:]

        prob[key] =  (str(int(num)+y_train[i]))+'/'+str(1+int(den))



    from fractions import Fraction

    for key in prob:
        prob[key] = float(Fraction(prob[key]))

    print(prob)

getFraction(4,4,train_data, test_data, dicts)