import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import yfinance as yf

from pandas_datareader import data as pdr
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM

# Load data of company
company = 'AAPL'  # ticker symbol

yf.pdr_override()

# From which point load data
start_date = dt.datetime(2018, 1, 1)
end_date = dt.datetime(2022, 2, 10)

data = pdr.get_data_yahoo(company, start_date, end_date)

# Data scaling from 0 to 1
scaler = MinMaxScaler(feature_range=(0, 1))
# predicting CLOSE price
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

how_many_days = 60

x_training = []
y_training = []

for x in range(how_many_days, len(scaled_data)):
    x_training.append(scaled_data[x - how_many_days:x, 0])
    y_training.append(scaled_data[x, 0])

x_training, y_training = np.array(x_training), np.array(y_training)
x_training = np.reshape(x_training, (x_training.shape[0], x_training.shape[1], 1))

# Use sequential model and add layers
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_training.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Prediction of the next day closing price

# Compile & train
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_training, y_training, epochs=26, batch_size=33)

""" Test the model accuracy on existing data """

# Load test data

start_test = dt.datetime(2022, 1, 1)
end_test = dt.datetime.now()

data_test = pdr.get_data_yahoo(company, start_test, end_test)
actual_prices = data_test['Close'].values  # historical values

# concatenate actual data and test data
all_dataset = pd.concat((data['Close'], data_test['Close']), axis=0)

model_input = all_dataset[len(all_dataset) - len(data_test) - how_many_days:].values
model_input = model_input.reshape(-1, 1)
model_input = scaler.transform(model_input)

# Make predictions on test data

x_test = []

for x in range(how_many_days, len(model_input)):
    x_test.append(model_input[x - how_many_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot the test predictions
plt.plot(actual_prices, color="black", label=f"Actual {company} price")
plt.plot(predicted_prices, color="red", label=f"Predicted {company} price")
plt.title(f"{company} Share price")
plt.xlabel('Time[days]')
plt.ylabel('Price[$]')
plt.legend()
plt.show()

# Predict next day

data_core = [model_input[len(model_input) + 1 - how_many_days: len(model_input + 1), 0]]
data_core = np.array(data_core)
data_core = np.reshape(data_core, (data_core.shape[0], data_core.shape[1], 1))

# Output
prediction = model.predict(data_core)  # Real data as input
prediction = scaler.inverse_transform(prediction)
print(f"Predicted price: {prediction}")
