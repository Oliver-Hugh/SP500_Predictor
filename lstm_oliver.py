import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

window_size = 5


def string_to_datetime(s):
    split = s.split('-')
    year = int(split[0])
    month = int(split[1])
    day = int(split[2])
    return datetime.datetime(year=year, month=month, day=day)


def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=window_size):
    first_date = string_to_datetime(first_date_str)
    last_date = string_to_datetime(last_date_str)
    target_date = first_date
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
    for j in range(n):
        X[:, j]
        ret_df[f'Target-{n - j}'] = X[:, j]
    ret_df['Target'] = Y
    return ret_df


def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()
    date_list = df_as_np[:, 0]
    middle_matrix = df_as_np[:, 1:-1]
    X = middle_matrix.reshape((len(date_list), middle_matrix.shape[1], 1))
    Y = df_as_np[:, -1]
    return date_list, X.astype(np.float32), Y.astype(np.float32)


sp_df = pd.read_csv('^SPX.csv')
sp_df = sp_df[['Date', 'Close']]
close_data = sp_df['Close']
dates = sp_df['Date']

size = len(close_data)
date_lst = []
close_lst = []
# Also get the close prices and put them into the close_lst
last_close = close_data.iloc[0]
bias = last_close  # the first value
for i in range(size):
    date_string = str(dates.iloc[i])
    date_string = date_string[0:10]
    date_lst.append(date_string)
    close = close_data.iloc[i]
    close_diff = close - bias
    close_percent_change = close_diff / bias
    last_close = close
    close_lst.append(close_percent_change)
# This is our newly formatted dataframe
sp_df = pd.DataFrame(data={'Date': date_lst, 'Close': close_lst})
# We want to format the dates from a string to datetime now, using our function
sp_df['Date'] = sp_df['Date'].apply(string_to_datetime)
sp_df.index = sp_df.pop('Date')
print(sp_df)

# Start day second time around: '2021-03-25'
windowed_df = df_to_windowed_df(sp_df, '2019-01-01', '2023-12-01')
dates, X, y = windowed_df_to_date_X_y(windowed_df)

# Calculate values to partition data set between training, validation, and testing
num_datapoints = int(len(dates))
training_proportion = .75
training_section = int(training_proportion * num_datapoints)
val_test_split = int(((1 - training_proportion) * .5 + training_proportion) * num_datapoints)

dates_train, X_train, y_train = dates[:training_section], X[:training_section], y[:training_section]
dates_val, X_val, y_val = (dates[training_section:val_test_split],
                           X[training_section:val_test_split], y[training_section:val_test_split])
dates_test, X_test, y_test = dates[val_test_split:], X[val_test_split:], y[val_test_split:]


model = Sequential([layers.Input((window_size, 1)),
                    layers.LSTM(50),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)])
model.compile(loss='mse',
              optimizer=Adam(learning_rate=0.001),
              metrics=['mean_absolute_error'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200)
train_predictions = model.predict(X_train).flatten()
val_predictions = model.predict(X_val).flatten()
test_predictions = model.predict(X_test).flatten()


# the following are the pseudo normalized values

plt.plot(dates_train, y_train)  # Training Observations
plt.plot(dates_val, y_val)      # Validation Observations
plt.plot(dates_test, y_test)    # Test Observations

plt.plot(dates_train, train_predictions)    # Training Predictions
plt.plot(dates_val, val_predictions)        # Validation Predictions
plt.plot(dates_test, test_predictions)      # Test Predictions
plt.legend(['Training Observations', 'Validation Observations', 'Test Observations',
            'Training Predictions', 'Validation Predictions', 'Test Predictions'])
plt.xlabel('Time')
plt.ylabel('(S&P 500 Closing Price - Bias)/Bias')
plt.title('Pseudo-Normalized Closing Price Prediction vs. Time')
plt.show()

classification_error = 0

train_predictions_actual = []
y_train_actual = []
train_error_lst = []
train_error_sum = 0
# for loop for training data
for a in range(len(dates_train)):
    # Reverse pseudo normalization
    prediction_actual = train_predictions[a] * bias + bias
    y_value = y_train[a] * bias + bias

    # Append to empty lists
    y_train_actual.append(y_value)
    train_predictions_actual.append(prediction_actual)

    # Compute error for metrics
    train_error = (prediction_actual - y_value) / y_value
    train_error_lst.append(train_error)
    train_error_sum = train_error_sum + abs(train_error)
    if a > 0:
        y_diff = y_value - y_train_actual[a-1]
        prediction_diff = prediction_actual - y_train_actual[a-1]
        if (prediction_diff < 0 and y_diff > 0) or (prediction_diff > 0 and y_diff < 0):
            classification_error += 1


val_predictions_actual = []
y_val_actual = []
val_error_lst = []
val_error_sum = 0
# for loop for validation data
for b in range(len(dates_val)):
    # Reverse pseudo normalization
    prediction_actual = val_predictions[b] * bias + bias
    y_value = y_val[b] * bias + bias

    # Append to empty lists
    val_predictions_actual.append(prediction_actual)
    y_val_actual.append(y_value)

    # Compute error for metrics
    val_error = (prediction_actual - y_value) / y_value
    val_error_lst.append(val_error)
    val_error_sum = val_error_sum + abs(val_error)
    # Check for classification error
    if b > 0:
        y_diff = y_value - y_val_actual[b-1]
        prediction_diff = prediction_actual - y_val_actual[b-1]
        if (prediction_diff < 0 and y_diff > 0) or (prediction_diff > 0 and y_diff < 0):
            classification_error += 1

# for loop for test data
test_predictions_actual = []
y_test_actual = []
test_error_lst = []
test_error_sum = 0
# for loop for validation data
for c in range(len(dates_test)):
    # Reverse pseudo normalization
    prediction_actual = test_predictions[c] * bias + bias
    y_value = y_test[c] * bias + bias

    # Append to empty lists
    test_predictions_actual.append(prediction_actual)
    y_test_actual.append(y_value)

    # Compute error for metrics
    test_error = (prediction_actual - y_value) / y_value
    test_error_lst.append(test_error)
    test_error_sum = test_error_sum + abs(test_error)
    # Check for classification error
    if c > 0:
        y_diff = y_value - y_test_actual[c-1]
        prediction_diff = prediction_actual - y_test_actual[c-1]
        if (prediction_diff < 0 and y_diff > 0) or (prediction_diff > 0 and y_diff < 0):
            classification_error += 1


# Plot the Actual stock value predictions
plt.figure()
plt.plot(dates_train, y_train_actual)  # Training Observations
plt.plot(dates_val, y_val_actual)      # Validation Observations
plt.plot(dates_test, y_test_actual)    # Test Observations

plt.plot(dates_train, train_predictions_actual)    # Training Predictions
plt.plot(dates_val, val_predictions_actual)        # Validation Predictions
plt.plot(dates_test, test_predictions_actual)      # Test Predictions
plt.legend(['Training Observations', 'Validation Observations', 'Test Observations',
            'Training Predictions', 'Validation Predictions', 'Test Predictions'])
plt.xlabel('Time')
plt.ylabel('S&P 500 Closing Price')
plt.title('Closing Price Prediction vs. Time')
plt.show()

avg_train_error = train_error_sum / len(train_error_lst)
avg_val_error = val_error_sum / len(val_error_lst)
avg_test_error = test_error_sum / len(test_error_lst)
print("Average Training Error is ", avg_train_error)
print("Average Validation Error is ", avg_val_error)
print("Average Test Error is ", avg_test_error)

print("Classification Errors: ", classification_error)
total_classification_error = classification_error/len(dates)
print("Total Classification Error is: ", total_classification_error)
print("This is a classification Accuracy of ", (1 - total_classification_error))

plt.legend(['Training Error', 'Validation Error', 'Test Error'])
plt.figure()
plt.plot(dates_train, train_error_lst)
plt.plot(dates_val, val_error_lst)
plt.plot(dates_test, test_error_lst)
plt.legend(['Training Error', 'Validation Error', 'Test Error'])
plt.ylabel("Error")
plt.xlabel("Time")
plt.title("Stock Price Observation Error")
plt.show()
