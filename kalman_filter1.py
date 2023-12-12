# import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
# import scipy
import financial_data
import math


def single_var_gauss(x, mu_gauss, sigma):
    factor = 1 / (sigma * math.sqrt(2 * math.pi))
    likelihood = factor * np.exp(-1/2 * (((x - mu_gauss) / sigma) ** 2))
    return likelihood


def update(mu1, mu2, var1, var2):
    mu_prime = (mu1 * var2 + var1 * mu2) / (var2 + var1)
    inv_var1 = 1 / var1
    inv_var2 = 1 / var2
    update_var = 1 / (inv_var1 + inv_var2)
    return [mu_prime, update_var]


def predict(mu_old, mu_motion, var_old, var_motion):
    mu_predict_prime = mu_old + mu_motion
    var_predict_prime = var_old + var_motion
    return [mu_predict_prime, var_predict_prime]

# Get the S&P500 data for the specified time frame and assign it to variable df
training_start = "2022-01-01"
# training_start = "2022-12-20"
training_end = "2022-03-01"
training_data_dataframe = financial_data.get_timeframe_data(training_start, training_end, False)

dates, data = financial_data.kalman_1_get_data(training_data_dataframe)
# We now have the date and the day

size = len(dates) - 1
plot_arr = np.zeros((1, size))
for i in range(size):
    plot_arr[0][i] = i
# Empty array to put predictions in, should be 1 less size b/c won't predict for the first day
predictions = np.zeros((1, size))

# initial parameters, arbitrary guesses
mu = 0.
sig = 1000.

# Biases, can vary this later
measurement_sig = 30.
motion_sig = 30.

close_data = []

for n in range(size):
    # measurement update, with uncertainty
    mu, sig = update(mu, sig, data[n][0], measurement_sig)
    print('Update: [{}, {}]'.format(mu, sig))
    # motion update, with uncertainty
    mu, sig = predict(mu, sig, data[n][1], motion_sig)
    close_data.append(data[n][0])
    predictions[0][n] = mu
    print('Predict: [{}, {}]'.format(mu, sig))


#plt.plot(plot_arr[0], predictions[0], 'b')
plt.plot(plot_arr[0], close_data[:size], 'r')

plt.show()
