# Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pandas_datareader import data as pdr

from sklearn.preprocessing import MinMaxScaler

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model
import streamlit as st

# Override pandas_datareader methods with yfinance
yf.pdr_override()

# Date range
start = '2013-01-01'
end = '2023-06-27'

# Streamlit interface
st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# Fetch the data
df = pdr.get_data_yahoo(user_input, start=start, end=end)

# Describing data
st.subheader('Data from 2013 - 2023')
st.write(df.describe())

# Visualisations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close, 'b')
st.pyplot(fig)


# 100 Moving Average
st.subheader('Closing Price vs Time Chart (with 100MA)')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(df.Close, 'b')
st.pyplot(fig)

# 200 Moving Average
st.subheader('Closing Price vs Time Chart (with 100MA & 200MA)')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)



# split data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

print(data_training.shape)
print(data_testing.shape)

scaler = MinMaxScaler(feature_range=(0,1))

# Scales the features of the 'data_training' dataset to the range [0, 1] and stores the scaled data as an array in 'data_training_array'
data_training_array = scaler.fit_transform(data_training)


# Load my nodel
model = load_model('keras_model.h5')


# Testing
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])


x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


# Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)
