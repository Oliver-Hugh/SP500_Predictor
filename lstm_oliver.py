import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import yfinance
import financial_data


def str_to_datetime(s):
    #s = str(s)
    #s = s[0:10]
    split = s.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year=year, month=month, day=day)


def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
    first_date = str_to_datetime(first_date_str)
    last_date = str_to_datetime(last_date_str)
    target_date = first_date
    #print(target_date)
    #print(type(target_date))
    dates = []
    X, Y = [], []
    last_time = False
    # print("error here", dataframe.loc[target_date])
    while True:
        df_subset = dataframe.loc[:target_date].tail(n + 1)
        if len(df_subset) != n + 1:
            print(f'Error: Window of size {n} is too large for date {target_date}')
            return
        values = df_subset['Close'].to_numpy()
        x, y = values[:-1], values[-1]
        dates.append(target_date)
        X.append(x)
        Y.append(y)
        next_week = dataframe.loc[target_date:target_date + datetime.timedelta(days=7)]
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split('T')[0]
        year_month_day = next_date_str.split('-')
        year, month, day = year_month_day
        next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))
        if last_time:
            break
        target_date = next_date
        if target_date == last_date:
            last_time = True
    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates
    X = np.array(X)
    for i in range(0, n):
        X[:, i]
        ret_df[f'Target-{n - i}'] = X[:, i]
    ret_df['Target'] = Y
    return ret_df


def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()
    dates = df_as_np[:, 0]
    middle_matrix = df_as_np[:, 1:-1]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))
    Y = df_as_np[:, -1]
    return dates, X.astype(np.float32), Y.astype(np.float32)


sp_df = pd.read_csv('^SPX.csv')
sp_df = sp_df[['Date', 'Close']]
close_data = sp_df['Close']


# start and end dates for all data (training, validation, and test)
#start = "2000-01-01"
#end = "2023-12-11"
#imported_df = financial_data.get_timeframe_data(start, end, False)

# The following code will take the data from the imported dataframe, and convert
# it to a new dataframe that is more convenient to use
#close_data = imported_df['Close']
size = len(close_data)
date_lst = []
close_pc_lst = []
last_close = close_data.iloc[0]
first_close = last_close
# Get the dates, as they are by default the index
# Also get the close prices and put them into the close_lst
"""for i in range(1, size):
    #date_string = str(imported_df.index[i])
    #date_string = str(sp_df.index[i])
    #date_string = date_string[0:10]
    #date_lst.append(date_string)
    
    close = close_data.iloc[i]
    close_diff = close - last_close
    close_percent_change = close_diff / first_close
    close_pc_lst.append(close_percent_change)
    last_close = close"""
# This is our newly formatted dataframe
#sp_df = pd.DataFrame(data={'Date': date_lst, 'Close': close_pc_lst})
# We want to format the dates from a string to datetime now, using our function
sp_df['Date'] = sp_df['Date'].apply(str_to_datetime)
sp_df.index = sp_df.pop('Date')
#plt.plot(sp_df.index, sp_df['Close'])
#plt.show()

# Start day second time around: '2021-03-25'
windowed_df = df_to_windowed_df(sp_df, '2021-03-25', '2022-03-23', n=3)
#print(windowed_df)
dates, X, y = windowed_df_to_date_X_y(windowed_df)
q_80 = int(len(dates) * .8)
q_90 = int(len(dates) * .9)
dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]
"""plt.plot(dates_train, y_train)
plt.plot(dates_val, y_val)
plt.plot(dates_test, y_test)
plt.legend(['Train', 'Validation', 'Test'])
#plt.show()"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
model = Sequential([layers.Input((3, 1)),
                    layers.LSTM(64),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)])
model.compile(loss='mse',
              optimizer=Adam(learning_rate=0.0001),
              metrics=['mean_absolute_error'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50)
train_predictions = model.predict(X_train).flatten()

plt.plot(dates_train, train_predictions)
plt.plot(dates_train, y_train)
plt.legend(['Training Predictions', 'Training Observations'])
plt.show()
