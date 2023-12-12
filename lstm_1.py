import tensorflow as tf
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# import scipy
import financial_data
import math


# Get the S&P500 data for the specified time frame and assign it to variable df
start = "2011-01-01"
end = "2023-11-01"
data_df = financial_data.get_timeframe_data(start, end, False)


def df_to_x_y(df, window_size):
    # input: [[-7 days closing, -6 days closing, ... , today closing]]
    # output: [5 days out (i+5) closing]
    df_as_np = df.to_numpy()
    x = []
    y = []
    for i in range(len(df_as_np) - window_size):
        # wrap each data point in its own list
        row = [[a.astype(np.float32)] for a in df_as_np[i:i+window_size]]
        x.append(row)
        true_value = df_as_np[i+window_size].astype(np.float32)
        y.append(true_value)

    return np.array(x), np.array(y)


data_input, data_output = df_to_x_y(data_df['Close'], 5)
print(data_input.shape)
print(data_output.shape)
print(data_output)

# There are 2973 data points. Training: 0 to 2000 Validation: 2000 to 2500 validation
# 2500 - present = test
x_train, y_train = data_input[:1500], data_output[:1500]
x_val, y_val = data_input[1500:2000], data_output[1500:2000]
x_test, y_test = data_input[2000:], data_output[2000:]
"""
print("y train is ", y_train)

print(x_train)
print(y_train)
plt.plot(y_train)
plt.show()
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

model1 = Sequential([layers.Input((5, 1)),
                    layers.LSTM(64),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)])
# model1.add(Dense(8, 'relu'))
model1.summary()

cp = ModelCheckpoint('model1/', save_best_only='True')
model1.compile(loss='mse',
               optimizer=Adam(learning_rate=0.001),
               metrics=['mean_absolute_error'])
model1.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5, callbacks=[cp])


from tensorflow.keras.models import load_model
model1 = load_model('model1/')

print(model1.predict(x_train))
train_predictions = model1.predict(x_train).flatten()
print(train_predictions)
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train})
plt.figure()
plt.plot(train_results['Train Predictions'], 'b')
plt.plot(train_results['Actuals'], 'r')
plt.show()

print(train_predictions)
