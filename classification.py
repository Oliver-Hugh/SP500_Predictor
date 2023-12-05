import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
import scipy
import financial_data


# Get the S&P500 data for the specified time frame and assign it to variable df
training_start = "2010-01-01"
training_end = "2023-01-01"
training_data_dataframe = financial_data.get_timeframe_data(training_start, training_end, False)
training_data = financial_data.convert_df_to_relative_array(training_data_dataframe)

daily_percent_increase = 0
inter_day_percent_increase = 1
normalized_volume = 2
p_range = 3
c_labels = 4

print(np.shape(training_data))


mean1 = np.mean(training_data[0][:])
mean2 = np.mean(training_data[1][:])
cov1 = np.cov(training_data[1][:])
cov2 = np.cov(training_data[1][:])

# Find within class scatter matrix
Sw = cov1 + cov2

# Between class scatter matrix
Sb = (mean1 - mean2) * np.transpose(mean1 + mean2)

# Compute LDA projection
#invSw = np.invert(Sw)
# invSw * Sb * w = J(w)w = lambda w, find eig
#invSw_by_Sb = invSw * Sb
# Matlab: [V, D] = eig(A, -B)
#D, V = np.linalg.eig(invSw_by_Sb)

# Projection Vector
#W = V[:][0]

#wt_x = np.transpose(W) * data

start_date = "2023-01-01"
num_days = 7


