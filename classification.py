import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
import scipy
import financial_data


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

# print(np.shape(training_data))
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
class_0_decrease = np.zeros((4, num_decrease))
class_1_increase = np.zeros((4, num_increase))
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

# Now we have our data for class 0 and class 1 - compute means and covariance matrices
mean_dec = np.mean(class_0_decrease, axis=1)
mean_inc = np.mean(class_1_increase, axis=1)
cov_dec = np.cov(class_0_decrease)
cov_inc = np.cov(class_1_increase)

training_data_minus_labels = training_data[0:4][:]

# Find within class scatter matrix
Sw = cov_dec + cov_inc

# Between class scatter matrix
Sb = (mean_dec - mean_inc) * np.transpose(mean_dec + mean_inc)

# Compute LDA projection
invSw = np.linalg.inv(Sw)
# invSw * Sb * w = J(w)w = lambda w, find eig
invSw_by_Sb = invSw * Sb
# Matlab: [V, D] = eig(A, -B)
D, V = np.linalg.eig(invSw_by_Sb)

# Projection Vector
W = np.zeros((1, 4))
W[0, :] = V[:, 0]

wt_x = np.matmul(W, training_data_minus_labels)
print(wt_x)
print(np.shape(wt_x))

# used np.min(wt_x) and np.max(wt_x) to get estimates for min and max to create linspace
n_gamma = 50
gamma = np.linspace(-.3, .05, n_gamma)
decisions = np.zeros((1, cols))
tpr_points = np.zeros((1, n_gamma))
fpr_points = np.zeros((1, n_gamma))

for z in range(n_gamma):
    thresh = gamma[z]
    true_pos_count = 0
    true_neg_count = 0
    false_pos_count = 0
    false_neg_count = 0
    num_correct = 0
    for y in range(1, cols):
        xx = training_data_minus_labels[:, y]
        classifier_ratio = wt_x[0][y]
        if classifier_ratio > thresh:
            decisions[0][y] = 0
        else:
            decisions[0][y] = 1
        if decisions[0][y] == int(training_data[c_labels][y]):
            num_correct += 1

        """
        # true or false positive (when L = 1 in reality)
        if int(training_data[c_labels][y]) == 1:
            if int(training_data[c_labels][y]) == decisions[0][y]:
                true_pos_count += 1
            else:
                false_neg_count += 1
        # true or false negative (when L = 0 in reality)
        else:
            if int(training_data[c_labels][y]) == decisions[0][y]:
                true_neg_count += 1
            else:
                false_pos_count += 1
    true_pos_rate = true_pos_count / (true_pos_count + false_neg_count)
    false_pos_rate = false_pos_count / (false_pos_count + true_neg_count)
    tpr_points[0][z] = true_pos_rate
    fpr_points[0][z] = false_pos_rate
    """
    tpr_points[0][z] = num_correct/cols
"""
min_error = 1
min_error_index = 0
for b in range(n_gamma):
    temp_error = .51 * fpr_points[0][b] + .49 * (1 - tpr_points[0][b])
    if temp_error < min_error:
        min_error = temp_error
        min_error_index = b

plt.figure()
plt.plot(fpr_points, tpr_points, '--ob')
plt.show()"""

print("the max correct is ", max(tpr_points[0][:]))
print(np.shape(tpr_points))
plt.figure()
plt.plot(gamma, tpr_points[0][:], 'ob')
plt.show()
print(np.shape(class_0_decrease))
print(np.shape(class_1_increase))
