import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

import matplotlib.pyplot as plt
import seaborn as sns


# ******************************************************
# *************** Data pre-processing*******************
# ******************************************************

stockData = open("MSFT2y.csv", "r")
indexData = open("^IXIC2y.csv", "r")

# generate dataframes out of the CSVs
df_stock = pd.read_csv(stockData, parse_dates=["Date"])
df_index = pd.read_csv(indexData, parse_dates=["Date"])

# edit and merge the stock dataframe
df_stock = df_stock[['Date','Close']]
df_index = df_index[['Date','Close']]
df_index = df_index.rename(columns={'Close' : 'Index Close'})
df = pd.merge(df_stock, df_index, how='inner', on='Date')
df = df.set_index('Date')[['Close', 'Index Close']]

# normalize all columns except the Date column
scaler = MinMaxScaler()
columnsWithNumbers = df.select_dtypes(np.number).columns
df[columnsWithNumbers] = scaler.fit_transform(df[columnsWithNumbers])

# here an uncommon step to split the data in a 95% train_test dataframe (later further split to train 75% & test data 25%) and
# a 5% 'past is future' dataframe which will be used at the end to test how well are 'real' future predictions doing
df_train_test, df_past_is_future = train_test_split(df, shuffle=False, test_size=0.05)

# generate X features and y targets with a window of 15 days --> why? look at 15 days to predict the 16th
df_input = df_train_test[['Close', 'Index Close']]

X = []
y = []
window = 15
for i in range(window, len(df_input)):
    X.append(df_input.iloc[i-window:i])
    y.append(df_input.iloc[i])
X=np.array(X)
y=np.array(y)

# generate the train and testing sets
X_train, X_test = train_test_split(X, shuffle=False, test_size=0.25)
y_train, y_test = train_test_split(y, shuffle=False, test_size=0.25)


# ******************************************************
# *************** Train, Test, Predict *****************
# ******************************************************

# Model training: Define and fit the model
closing_price_model = tf.keras.Sequential()
closing_price_model.add(tf.keras.layers.LSTM(500, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], 2)))
closing_price_model.add(tf.keras.layers.Dropout(0.2))
closing_price_model.add(tf.keras.layers.LSTM(200, activation='tanh', return_sequences=False))
closing_price_model.add(tf.keras.layers.Dropout(0.2))
closing_price_model.add(tf.keras.layers.Dense(2, activation='linear'))
closing_price_model.compile(loss = 'MSE', metrics=['MAE', 'MSE', 'MAPE'], optimizer='Adam')
closing_price_model.summary()
closing_price_model.fit(X_train, y_train, epochs=50)

# Model testing: Predict values (stock closing prices) based on test data containing real historical features
y_train_predict = closing_price_model.predict(X_train)
predicted_values_on_test_data = closing_price_model.predict(X_test)
test_results = pd.DataFrame({'Predicted_Close': predicted_values_on_test_data[:, 0], 'Predicted_Index Close': predicted_values_on_test_data[:, 1]})
# assign proper Date index to predicted values
test_results.index = df_train_test.index[-len(y_test):]

# Make 'real' future predictions after the model has been trained and tested
# take as initial starting point the last sequence of test data to predict the first 'real' future value
# this future value will become the new last element of the sequence, while the 1st sequence value drops out
# this process will be repeated in this scenario for the length of the past_is_future dataframe
# the future values will then be benchmarked against the df_past_is_future values
last_test_features_seq = [] 
last_test_features_seq.append(df_input[len(df_input)-window:])
input_buffer = np.array(last_test_features_seq)

j = 0 
while j < len(df_past_is_future.index):
    future_price_element = closing_price_model.predict(input_buffer)
    input_buffer = np.delete(input_buffer, 0, axis=1)
    input_buffer = np.append(input_buffer, [[future_price_element[0]]], axis=1)
    if j == 0:
        y_future = np.array(future_price_element)
    else:
        y_future = np.append(y_future, future_price_element, axis=0)
    j = j + 1

# after future closing prices have been predicted along future index closing in a nested array,
# the closing prices have to be separated for further use 
k = 0
while k < len(df_past_is_future.index):
    if k == 0:
        y_future_price = np.array(y_future[k][0])
        y_future_index = np.array(y_future[k][1])
    else:
        y_future_price = np.append(y_future_price, y_future[k][0])
        y_future_index = np.append(y_future_index, y_future[k][1])
    k= k+1

df_future = pd.DataFrame({'Future_Close': y_future_price, 'Future_Index Close': y_future_index})
# assign proper Date index to predicted values
df_future.index = df_past_is_future.index[0:len(y_future_price)]

# inverse scaling for MAPE calculation, because if actual values contain even one zero the 
# calculation returns high values as there will be a division through zero
df_y_test = pd.DataFrame(y_test)
df_y_test = scaler.inverse_transform(df_y_test)
df_y_test_predict = pd.DataFrame(predicted_values_on_test_data)
df_y_test_predict = scaler.inverse_transform(df_y_test_predict)


# ******************************************************
# ********************* Evaluation *********************
# ******************************************************

# Mean Absolute Percentage Error
mape_train_score = mean_absolute_percentage_error(y_train[:,0], y_train_predict[:,0])
mape_test_score = mean_absolute_percentage_error(df_y_test, df_y_test_predict)
mape_future_score = mean_absolute_percentage_error(df_past_is_future.Close, df_future.Future_Close)

# Mean Absolute Error
mae_train_score = mean_absolute_error(y_train[:,0], y_train_predict[:,0])
mae_test_score = mean_absolute_error(y_test[:,0], predicted_values_on_test_data[:,0])
mae_future_score = mean_absolute_error(df_past_is_future.Close, df_future.Future_Close)

# Mean Squared Error
mse_train_score = mean_squared_error(y_train[:,0], y_train_predict[:,0])
mse_test_score = mean_squared_error(y_test[:,0], predicted_values_on_test_data[:,0])
mse_future_score = mean_squared_error(df_past_is_future.Close, df_future.Future_Close)

# Root Mean Squared Error
rmse_train_score = np.sqrt(mean_squared_error(y_train[:,0], y_train_predict[:,0]))
rmse_test_score = np.sqrt(mean_squared_error(y_test[:,0], predicted_values_on_test_data[:,0]))
rmse_future_score = np.sqrt(mean_squared_error(df_past_is_future.Close, df_future.Future_Close))

scores = [['Scores', "MAPE", "MAE", "MSE", "RMSE"],
['Train', mape_train_score, mae_train_score, mse_train_score, rmse_train_score],
['Test', mape_test_score, mae_test_score, mse_test_score, rmse_test_score],
['Future', mape_future_score, mae_future_score, mse_future_score, rmse_future_score]]
print(pd.DataFrame(scores))


# ******************************************************
# ********************* Visualization ******************
# ******************************************************

sns.set()
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the entire Data set after inverse scaling
columnsWithNumbers = df.select_dtypes(np.number).columns
df[columnsWithNumbers] = scaler.inverse_transform(df[columnsWithNumbers])
plt.plot(df[['Close']], label=['Real_Close'])

# Plot the test data prediction with real feature data after inverse scaling
columnsWithNumbers = test_results.select_dtypes(np.number).columns
test_results[columnsWithNumbers] = scaler.inverse_transform(test_results[columnsWithNumbers])
plt.plot(test_results[['Predicted_Close']], label=['Predicted_Close on test data'], linestyle='--')
plt.axvspan(test_results.index[0], test_results.index[-1], facecolor='lightgreen', alpha=0.25, label='Test data prediction')

# Plot the 'real' predicted future prices based on forecasted (generated) feature data after inverse scaling
columnsWithNumbers = df_future.select_dtypes(np.number).columns
df_future[columnsWithNumbers] = scaler.inverse_transform(df_future[columnsWithNumbers])
plt.plot(df_future[['Future_Close']], label=['Future_Close on generated features'], linestyle='-.')
plt.axvspan(df_future.index[0], df_future.index[-1], facecolor='lightblue', alpha=0.25, label='Real future price prediction')

ax.legend()

plt.title('LSTM - Multivariate Stock Price Forecasting', pad = 50)
plt.table(cellText=scores, loc = 'top')
plt.xlabel('Date')
plt.ylabel('Price')

plt.show()
