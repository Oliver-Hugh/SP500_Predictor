import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
import scipy
import financial_data
import math

dimensions = 4


def single_var_gauss(x, mu, sigma):
    factor = 1 / (sigma * math.sqrt(2 * math.pi))
    likelihood = factor * np.exp(-1/2 * (((x - mu) / sigma) ** 2))
    return likelihood


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

decisions = np.zeros((1, cols))
num_correct = 0
y = np.zeros((1, dimensions))
for z in range(cols-1):
    y[0, :] = training_data_minus_labels[:, z]
    # Assume Gaussian
    # Calculate likelihoods for decrease
    daily_percent_increase_dec_likelihood = single_var_gauss(
        y[0, daily_percent_increase], mean_dec[daily_percent_increase],
        cov_dec[daily_percent_increase][daily_percent_increase])
    inter_day_percent_increase_dec_likelihood = single_var_gauss(
        y[0, inter_day_percent_increase], mean_dec[inter_day_percent_increase],
        cov_dec[inter_day_percent_increase][inter_day_percent_increase])
    normalized_volume_dec_likelihood = single_var_gauss(
        y[0, normalized_volume], mean_dec[normalized_volume],
        cov_dec[normalized_volume][normalized_volume])
    p_range_dec_likelihood = single_var_gauss(
        y[0, p_range], mean_dec[p_range], cov_dec[p_range][p_range])
    dec_likelihood = (daily_percent_increase_dec_likelihood *
                      inter_day_percent_increase_dec_likelihood *
                      normalized_volume_dec_likelihood * p_range_dec_likelihood * dec_prior)

    # Calculate likelihoods for increase
    daily_percent_increase_inc_likelihood = single_var_gauss(
        y[0, daily_percent_increase], mean_inc[daily_percent_increase],
        cov_inc[daily_percent_increase][daily_percent_increase])
    inter_day_percent_increase_inc_likelihood = single_var_gauss(
        y[0, inter_day_percent_increase], mean_inc[inter_day_percent_increase],
        cov_inc[inter_day_percent_increase][inter_day_percent_increase])
    normalized_volume_inc_likelihood = single_var_gauss(
        y[0, normalized_volume], mean_inc[normalized_volume],
        cov_inc[normalized_volume][normalized_volume])
    p_range_inc_likelihood = single_var_gauss(
        y[0, p_range], mean_inc[p_range], cov_inc[p_range][p_range])
    inc_likelihood = (daily_percent_increase_inc_likelihood *
                      inter_day_percent_increase_inc_likelihood *
                      normalized_volume_inc_likelihood * p_range_inc_likelihood * dec_prior)
    # Decisions
    if inc_likelihood > dec_likelihood:
        decisions[0][z] = 1
    else:
        decisions[0][z] = 0
    if decisions[0][z] == int(training_data[c_labels][z]):
        num_correct += 1
success = num_correct / cols
print("Success: ", success)

print("Prior ", inc_prior)
print(np.shape(class_0_decrease))
print(np.shape(class_1_increase))



print("Test Data Below")
# Get the S&P500 data for the specified time frame and assign it to variable df
test_start = "2022-01-01"
test_end = "2023-01-01"
test_data_dataframe = financial_data.get_timeframe_data(test_start, test_end, False)
test_data = financial_data.convert_df_to_relative_array(test_data_dataframe)


rows, cols = np.shape(test_data)
decisions = np.zeros((1, cols))
num_increase = 0
num_decrease = 0
num_correct = 0
xx = np.zeros((1, dimensions))

for i in range(cols):
    # If the day is a Decrease
    if test_data[c_labels][i] == 0:
        num_decrease += 1
    # If the day is an increase
    else:
        num_increase += 1

class_0_decrease = np.zeros((dimensions, num_decrease))
class_1_increase = np.zeros((dimensions, num_increase))

num_increase = -1
num_decrease = -1
for i in range(cols):
    # If the day is a Decrease
    if test_data[c_labels][i] == 0:
        num_decrease += 1
        class_0_decrease[:, num_decrease] = test_data[0:dimensions, i]
    # If the day is an increase
    else:
        num_increase += 1
        class_1_increase[:, num_increase] = test_data[0:4, i]
test_data_minus_labels = test_data[0:dimensions][:]
print("num decrease is ", num_decrease)
print("num increase is ", num_increase)

for z in range(cols-1):
    xx[0, :] = test_data_minus_labels[:, z]
    # Assume Gaussian
    # Calculate likelihoods for decrease
    daily_percent_increase_dec_likelihood = single_var_gauss(
        xx[0, daily_percent_increase], mean_dec[daily_percent_increase],
        cov_dec[daily_percent_increase][daily_percent_increase])
    inter_day_percent_increase_dec_likelihood = single_var_gauss(
        xx[0, inter_day_percent_increase], mean_dec[inter_day_percent_increase],
        cov_dec[inter_day_percent_increase][inter_day_percent_increase])
    normalized_volume_dec_likelihood = single_var_gauss(
        xx[0, normalized_volume], mean_dec[normalized_volume],
        cov_dec[normalized_volume][normalized_volume])
    p_range_dec_likelihood = single_var_gauss(
        xx[0, p_range], mean_dec[p_range], cov_dec[p_range][p_range])
    dec_likelihood = (daily_percent_increase_dec_likelihood *
                      inter_day_percent_increase_dec_likelihood *
                      normalized_volume_dec_likelihood * p_range_dec_likelihood * dec_prior)

    # Calculate likelihoods for increase
    daily_percent_increase_inc_likelihood = single_var_gauss(
        xx[0, daily_percent_increase], mean_inc[daily_percent_increase],
        cov_inc[daily_percent_increase][daily_percent_increase])
    inter_day_percent_increase_inc_likelihood = single_var_gauss(
        xx[0, inter_day_percent_increase], mean_inc[inter_day_percent_increase],
        cov_inc[inter_day_percent_increase][inter_day_percent_increase])
    normalized_volume_inc_likelihood = single_var_gauss(
        xx[0, normalized_volume], mean_inc[normalized_volume],
        cov_inc[normalized_volume][normalized_volume])
    p_range_inc_likelihood = single_var_gauss(
        xx[0, p_range], mean_inc[p_range], cov_inc[p_range][p_range])
    inc_likelihood = (daily_percent_increase_inc_likelihood *
                      inter_day_percent_increase_inc_likelihood *
                      normalized_volume_inc_likelihood * p_range_inc_likelihood * dec_prior)
    # Decisions
    if inc_likelihood > dec_likelihood:
        decisions[0][z] = 1
    else:
        decisions[0][z] = 0
    if decisions[0][z] == int(test_data[c_labels][z]):
        num_correct += 1

success = num_correct / cols
print("Success: ", success)
