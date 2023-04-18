import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data
from keras.models import load_model
import streamlit as st
import yfinance as yfin
from datetime import datetime
yfin.pdr_override()

start = datetime(2010,1,1)
end = datetime(2022,3,31)

st.title('Stock Trend Prediction')
user_input = st.text_input('Enter stock Ticker')
if(user_input):
    df = data.get_data_yahoo(user_input, start, end)
    # df = data.get_data_yahoo('TSLA', start, end)
    
    # Describing data
    st.subheader('data from 2010 to 2019')
    st.write(df.describe)


    # df.head()
    # df.tail()

    df = df.reset_index()
    df.head()

    df = df.drop(['Date', 'Adj Close'], axis = 1)
    plt.plot(df.Close)

    # 100 day moving average
    ma100 = df.Close.rolling(100).mean()
    # ma100

    # 200 day moving average
    ma200 = df.Close.rolling(200).mean()
    # ma200

    plt.figure(figsize = (12, 6))
    plt.plot(df.Close)
    plt.plot(ma100, 'r')
    plt.plot(ma200, 'g')

    # df.shape

    # Splitting data into training and testing
    data_training = pd.DataFrame(df['Close'][0 : int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70) : int(len(df))])

    print(data_training.shape)
    print(data_testing.shape)


    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range = (0,1))

    data_training_array = scaler.fit_transform(data_training)

    x_train = []
    y_train = []

    for i in range(100, data_training_array.shape[0]):
        x_train.append(data_training_array[i-100:i])
        y_train.append(data_training_array[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    #ML model
    from keras.layers import Dense, Dropout, LSTM
    from keras.models import Sequential

    model = Sequential()
    model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (x_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
    model.add(Dropout(0.3))

    model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
    model.add(Dropout(0.4))

    model.add(LSTM(units = 120, activation = 'relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units = 1))

    model.summary()

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    model.fit(x_train, y_train, epochs = 5)

    model.save('keras_model.h5')

    data_testing.head()

    past_100_days = data_training.tail(100)

    final_df = past_100_days.append(data_testing, ignore_index = True)
    final_df.head()

    input_data = scaler.fit_transform(final_df)
    input_data

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100 : i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)

    y_predicted.shape

    y_test
    y_predicted

    scale_factor = 1/scaler.scale_
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    plt.figure(figsize = (12, 6))
    plt.plot(y_test, 'b', label = 'Original Price')
    plt.plot(y_predicted, 'r', label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()