import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
import scipy
import financial_data
import math

dimensions = 4

# Get the S&P500 data for the specified time frame and assign it to variable df
training_start = "2012-01-01"
#training_start = "2022-12-20"
training_end = "2022-01-01"
training_data_dataframe = financial_data.get_timeframe_data(training_start, training_end, False)
training_data = financial_data.convert_df_to_relative_array(training_data_dataframe)

daily_percent_increase = 0
inter_day_percent_increase = 1
normalized_volume = 2
p_range = 3
c_labels = 4

rows, cols = np.shape(training_data)

num_increase = 0
num_decrease = 0
for i in range(cols):
    # If the day is a Decrease
    if training_data[c_labels][i] == 0:
        num_decrease += 1
    # If the day is an increase
    else:
        num_increase += 1
class_0_decrease = np.zeros((dimensions, num_decrease))
class_1_increase = np.zeros((dimensions, num_increase))

inc_prior = num_increase / cols
dec_prior = num_decrease / cols
num_increase = -1
num_decrease = -1
for i in range(cols):
    # If the day is a Decrease
    if training_data[c_labels][i] == 0:
        num_decrease += 1
        class_0_decrease[:, num_decrease] = training_data[0:4, i]
    # If the day is an increase
    else:
        num_increase += 1
        class_1_increase[:, num_increase] = training_data[0:4, i]
training_data_minus_labels = training_data[0:dimensions][:]


# Now we have our data for class 0 and class 1 - compute means and covariance matrices
mean_dec = np.mean(class_0_decrease, axis=1)
mean_inc = np.mean(class_1_increase, axis=1)
cov_dec = np.cov(class_0_decrease)
cov_inc = np.cov(class_1_increase)
inv_cov_dec = np.linalg.inv(cov_dec)
inv_cov_inc = np.linalg.inv(cov_inc)

xx = np.zeros((1, dimensions))
xx[0, :] = training_data_minus_labels[:, 0]

n_gamma = 100
gamma = np.logspace(-2, 4, n_gamma)
decisions = np.zeros((1, cols))
success = np.zeros((1, n_gamma))


for z in range(n_gamma):
    thresh = gamma[z]
    num_correct = 0
    for y in range(cols-1):
        xx = training_data_minus_labels[:, y]

        # Assume Gaussian

        # Decrease gaussian likelihood
        decrease_gaussian_factor = (1 / (((2 * math.pi) ** (dimensions / 2)) * math.sqrt(np.linalg.det(cov_dec))))
        diff_dec = xx - mean_dec
        factor_dec = np.matmul(diff_dec, inv_cov_dec)
        factor_dec = (-1 / 2 * np.matmul(factor_dec, np.transpose(diff_dec)))#[0][0]
        decrease_likelihood = decrease_gaussian_factor * np.exp(factor_dec)

        # Increase gaussian likelihood
        increase_gaussian_factor = (1 / (((2 * math.pi) ** (dimensions / 2)) * math.sqrt(np.linalg.det(cov_inc))))
        diff_inc = xx - mean_inc
        factor_inc = np.matmul(diff_inc, inv_cov_inc)
        factor_inc = (-1 / 2 * np.matmul(factor_inc, np.transpose(diff_inc)))#[0][0]
        increase_likelihood = increase_gaussian_factor * np.exp(factor_inc)

        # P[x | L = 1] / P[x | L = 0]
        classifier_ratio = increase_likelihood / decrease_likelihood
        if classifier_ratio > thresh:
            decisions[0][y] = 0
        else:
            decisions[0][y] = 1
        if decisions[0][y] == int(training_data[c_labels][y]):
            num_correct += 1
    success[0][z] = num_correct/cols

print("Max success rate is ", max(success[0][:]), " at a threshold of ", gamma[np.argmax(success[0][:])])
plt.figure()
plt.plot(gamma, success[0][:], 'ob')
plt.xlabel("Threshold")
plt.ylabel("Classification Success Rate")
plt.title("Maximum Likelihood: S&P500 Increase/Decrease Classifier")
plt.show()
print(inc_prior)
print(np.shape(class_0_decrease))
print(np.shape(class_1_increase))
