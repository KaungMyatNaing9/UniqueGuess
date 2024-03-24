#An explanation of the code is provided below (Copilot generated)
#This Python script is used to fetch historical stock data, preprocess it, and set up a Long Short-Term Memory (LSTM) model for predicting future stock prices.
#
#The script begins by fetching historical data for a specific stock, in this case, Nvidia (NVDA), using the yf.download() function from the yfinance library. The data is fetched for a specific date range, from the start of 2020 to March 2024.
#
#Next, the script isolates the 'Close' prices from the fetched data, as these are the values that will be used for prediction.
#
#The data is then normalized using the MinMaxScaler from the sklearn.preprocessing library. This scales the 'Close' prices to a range between 0 and 1, which can help improve the performance of the LSTM model later on.
#
#The script then creates a training data set, which consists of 95% of the total data. The remaining 5% could be used as a test set to evaluate the model's performance.
#
#The training data is then split into x_train and y_train data sets. For each point in the training data, the previous 60 points are used as input features (x_train), and the current point is used as the output label (y_train). This means the model will be trained to predict the current stock price based on the previous 60 prices.
#
#The x_train data is then reshaped into a 3D array, which is the input shape expected by the LSTM model.
#
#Finally, the script starts building the LSTM model using the Sequential model from the keras library. The model consists of an LSTM layer with 50 units, followed by a Dropout layer for regularization (to prevent overfitting), and another LSTM layer with 50 units. The return_sequences parameter is set to True for the first LSTM layer because the next layer is also an LSTM layer. For the final LSTM layer, return_sequences is set to False because there are no more LSTM layers following it.
#
#The model is then compiled with the Adam optimizer and the mean squared error loss function, and it's trained on the training data set.
#
#After training, the model is used to predict the stock prices for the test set. The last 60 days of the scaled data are used as the initial input sequence for the predictions. The predicted prices are then inverse transformed to get the actual price predictions.
#
#Finally, the script plots the historical and predicted prices on a line chart, with the date on the x-axis and the price on the y-axis. The historical prices are shown in blue, and the predicted prices are shown in orange.
######################################################################################################################################

import numpy as np  #calculations
import pandas as pd #data manipulation
import yfinance as yf #fetching data
from sklearn.preprocessing import MinMaxScaler #scaling data
from tensorflow.keras.models import Sequential  #model
from tensorflow.keras.layers import Dense, LSTM, Dropout #layers
import matplotlib.pyplot as plt #plotting
from datetime import datetime

#Fetching Data
ticker_symbol = input("Enter the ticker symbol(ex:APPL): ")
data = yf.download(ticker_symbol, start='2020-01-01', end='2024-03-24')

# Use only the 'Close' price for prediction
close_prices = data[['Close']]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_prices)

# Create a training data set
train_data_len = int(np.ceil( len(scaled_data) * .95 ))

# Create the scaled training data set
train_data = scaled_data[0:int(train_data_len), :]

# Split the data into x_train and y_train data sets
x_train, y_train = [], []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data into the shape accepted by the LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM network model(credit:GPT)
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Test data set
test_data = scaled_data[train_data_len - 60: , :]

# Create the x_test and y_test data sets
x_test = []
y_test = close_prices.iloc[train_data_len:, :].values

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the model's predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Plot the data
train = close_prices[:train_data_len]
valid = close_prices[train_data_len:]
valid['Predictions'] = predictions

# Visualize the model's performance
plt.figure(figsize=(16,8))
plt.title(ticker_symbol)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Valid', 'Prediction'], loc='lower right')
plt.show()

