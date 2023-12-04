import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
import scipy


def get_data_for_timeframe(start_date, end_date):
    try:
        # Fetch S&P500 from January 1, 2010, to today
        start_date_full = "2010-01-01"
        end_date_full = "2023-12-31"
        full_data = yf.download("^GSPC", start_date_full, end_date_full)

        # Filter data within the specified time frame
        df = full_data.loc[start_date : end_date]
        # Plot closing prices
        """
        df['Close'].plot(figsize=(10, 6), title='S&P500 Index')
        plt.xlabel('Date')
        plt.ylabel('Closing Price (USD)')
        plt.show()
        """
        return df
    except Exception as e:
        print(f"Error:{e}")
        return None


# Get the S&P500 data for the specified time frame and assign it to variable df
df = get_data_for_timeframe("2022-01-01", "2022-01-02")
print(df)
starting_point = 0
N = 100
# Array to place data in to
data = np.zeros((2, 10), float)

mean1 = np.mean(data[0][:])
mean2 = np.mean(data[1][:])
cov1 = np.cov(data[1][:])
cov2 = np.cov(data[1][:])

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


