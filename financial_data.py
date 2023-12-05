import yfinance as yf
from matplotlib import pyplot as plt
import numpy as np


def get_timeframe_data(start_date, end_date, graph_on):
    try:
        # Fetch S&P500 from January 1, 2010, to today
        start_date_full = "2010-01-01"
        end_date_full = "2023-12-31"
        full_data = yf.download("^GSPC", start_date_full, end_date_full)

        # Filter data within the specified time frame
        df = full_data.loc[start_date:end_date]
        # Plot closing prices
        if graph_on:
            df['Close'].plot(figsize=(10, 6), title='S&P500 Index')
            plt.xlabel('Date')
            plt.ylabel('Closing Price (USD)')
            plt.show()
        return df
    except Exception as e:
        print(f"Error:{e}")
        return None


def convert_df_to_relative_array(data_frame):
    # Column index. We will generally just use adj_close_price instead of close_price
    open_price = 0
    high_price = 1
    low_price = 2
    # close_price = 3
    adj_c_price = 4
    volume = 5
    orig_data = data_frame.to_numpy()
    n = np.shape(orig_data)[0]
    usable_array = np.zeros((n-1, 5))

    vol_arr = np.transpose(orig_data[:, 5])
    vol_max = vol_arr.max()

    for i in range(1, n):
        # Percent increase from open to close
        daily_percent_increase = (orig_data[i][adj_c_price] - orig_data[i][open_price])/orig_data[i][open_price]
        usable_array[i-1][0] = daily_percent_increase

        # Percent increase from day n to day n - 1
        inter_day_percent_increase = (orig_data[i][adj_c_price] -
                                      orig_data[i-1][adj_c_price])/orig_data[i-1][adj_c_price]
        usable_array[i-1][1] = inter_day_percent_increase

        # Normalize the volume
        normalized_volume = orig_data[i][volume] / vol_max
        usable_array[i-1][2] = normalized_volume

        # Price Range
        p_range = orig_data[i][high_price] - orig_data[i][low_price]
        usable_array[i-1][3] = p_range

    for j in range(1, n - 1):
        next_daily_percent_increase = usable_array[j][0]
        # Class Label: Decrease (0) or Increase (1)
        if next_daily_percent_increase >= 0:
            usable_array[j][4] = 1
        else:
            usable_array[j][4] = 0
    """
    daily_percent_increase = 0
    inter_day_percent_increase = 1
    normalized_volume = 2
    p_range = 3
    c_labels = 4
    """
    return np.transpose(usable_array)


def lda2_data(data_frame):
    # Column index. We will generally just use adj_close_price instead of close_price
    open_price = 0
    high_price = 1
    low_price = 2
    # close_price = 3
    adj_c_price = 4
    volume = 5
    orig_data = data_frame.to_numpy()
    n = np.shape(orig_data)[0]
    usable_array = np.zeros((n-1, 4))

    vol_arr = np.transpose(orig_data[:, 4])
    vol_max = vol_arr.max()

    for i in range(1, n-1):
        # Percent increase from open to close
        daily_percent_increase = (orig_data[i][adj_c_price] - orig_data[i][open_price])/orig_data[i][open_price]
        usable_array[i-1][0] = daily_percent_increase

        # Percent increase from day n to day n - 1
        inter_day_percent_increase = (orig_data[i][adj_c_price] -
                                      orig_data[i-1][adj_c_price])/orig_data[i-1][adj_c_price]
        usable_array[i-1][1] = inter_day_percent_increase

        # Normalize the volume
        normalized_volume = orig_data[i][volume] / vol_max
        usable_array[i-1][2] = normalized_volume
    for j in range(1, n-1):
        next_daily_percent_increase = usable_array[j][0]
        # Class Label: Decrease (0) or Increase (1)
        if next_daily_percent_increase >= 0:
            usable_array[j][3] = 1
        else:
            usable_array[j][3] = 0
    """
    daily_percent_increase = 0
    inter_day_percent_increase = 1
    normalized_volume = 2
    c_labels = 3
    """
    return np.transpose(usable_array)


def lda3_data(data_frame):
    # Column index. We will generally just use adj_close_price instead of close_price
    open_price = 0
    high_price = 1
    low_price = 2
    # close_price = 3
    adj_c_price = 4
    volume = 5
    orig_data = data_frame.to_numpy()
    n = np.shape(orig_data)[0]
    usable_array = np.zeros((n-1, 7))

    vol_arr = np.transpose(orig_data[:, 4])
    vol_max = vol_arr.max()

    for i in range(2, n):
        # Percent increase from open to close
        daily_percent_increase = (orig_data[i][adj_c_price] - orig_data[i][open_price])/orig_data[i][open_price]
        usable_array[i-2][0] = daily_percent_increase
        usable_array[i-1][1] = (orig_data[i-1][adj_c_price] - orig_data[i-1][open_price])/orig_data[i-1][open_price]

        # Percent increase from day n to day n - 1
        inter_day_percent_increase = (orig_data[i][adj_c_price] -
                                      orig_data[i-1][adj_c_price])/orig_data[i-1][adj_c_price]
        usable_array[i-1][2] = inter_day_percent_increase
        usable_array[i - 2][3] = (orig_data[i-1][adj_c_price] -
                                  orig_data[i-2][adj_c_price])/orig_data[i-2][adj_c_price]

        # Normalize the volume
        normalized_volume = orig_data[i][volume] / vol_max
        usable_array[i-1][4] = normalized_volume

        # Price Range
        p_range = orig_data[i][high_price] - orig_data[i][low_price]
        usable_array[i-1][5] = p_range


    for j in range(1, n-1):
        next_daily_percent_increase = usable_array[j][0]
        # Class Label: Decrease (0) or Increase (1)
        if next_daily_percent_increase >= 0:
            usable_array[j][6] = 1
        else:
            usable_array[j][6] = 0
    """
    daily_percent_increase = 0
    yesterday_percent_increase = 1
    inter_day_percent_increase = 2
    inter_day_yesterday_percent_increase = 3
    normalized_volume = 4
    p_range = 5
    c_labels = 6
    """
    return np.transpose(usable_array)