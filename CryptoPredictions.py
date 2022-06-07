import streamlit as st
import datetime as dt
from plotly import graph_objs as go
import numpy as np
import pandas as pd
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from yahoo_fin import stock_info as si
import time


@st.cache
def load_data(ticker):
    return web.DataReader(ticker, 'yahoo', START, TODAY)


def predict(selected_coin, period):
    data_load_state.text("Loading Data for " + selected_coin)
    print(selected_coin)
    data = load_data(selected_coin)
    raw_data = data.tail()

    data_load_state.text("Processing Data for " + selected_coin + ": Formatting the Data")
    # Format Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    prediction_days = 10
    x_train = []
    y_train = []
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the Model
    data_load_state.text("Processing Data for " + selected_coin + ": Building the Model (This will take a second!)")
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Prediction of next price
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=250, batch_size=32)  # Mess with epochs and batch size

    # Test the Model on Past Performance
    data_load_state.text("Processing Data: Testing the Model")
    test_start = dt.datetime(2021, 1, 1)
    test_end = dt.datetime.now()
    test_data = web.DataReader(selected_coin, 'yahoo', test_start, test_end)
    actual_prices = test_data['Close'].values
    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    # Make Predictions on Test Data
    data_load_state.text("Processing Data: Making Predictions Using the Model")
    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    plot_predictions = []
    for i in range(len(predicted_prices)):
        plot_predictions.append(float(predicted_prices[i]))

    # Plot the Predictions
    data_load_state.text("Processing Data for " + selected_coin + ": Plotting the Past Predictions")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(y=plot_predictions, name='Predicted Closing Prices for ' + selected_coin))
    fig1.add_trace(go.Scatter(y=actual_prices, name='Actual Closing Prices for ' + selected_coin))
    fig1.layout.update(title_text="Predictions of Past Prices", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig1)

    # Predict the Future
    data_load_state.text("Processing Data: Predicting the Future")
    # plot_predictions = []
    real_data = [model_inputs[len(model_inputs) + period - prediction_days:len(model_inputs) + period, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    predictions = model.predict(real_data)
    predictions = scaler.inverse_transform(predictions)
    stri = "Prediction of " + str(selected_coin)
    stri += " " + str(int(period))
    stri += " Days into the Future: $" + str(float(predictions[0]))
    # plot_predictions.append(float(predictions[0]))
    # fig2 = go.Figure()
    # fig2.add_trace(go.Scatter(y=plot_predictions, name='Predicted Closing Prices'))
    # fig2.layout.update(title_text="Predictions of Future Prices", xaxis_rangeslider_visible=True)
    # st.plotly_chart(fig2)

    # Raw Data
    data_load_state.text("Processing Data for " + selected_coin + ": Done \n" + stri)
    st.subheader('Raw Data Snippet for ' + selected_coin)
    st.write(raw_data)
    return [predictions[0], plot_predictions[len(plot_predictions)-1]]


START = dt.datetime(2021, 1, 1)
TODAY = dt.datetime.now()
st.title("Crypto Predictions Using Neural Networks")
st.text("Web App by Shreyas Pani")
coins = ("BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "LTC-USD", "XLM-USD", "SHIB-USD", "ZRX-USD", "1INCH-USD", "AAVE-USD", "ACH-USD", "ALGO-USD", "AMP1-USD", "ANKR-USD", "REP-USD", "AVAX-USD", "AXS-USD", "BAL-USD", "BNT-USD", "BAND-USD", "BAT-USD", "BCH-USD", "CTSI-USD", "CELO-USD", "LINK-USD", "CHZ-USD", "CVC-USD", "CLV-USD", "COMP-USD", "ATOM1-USD", "COTI-USD", "CRV-USD", "DAI1-USD", "DASH-USD", "MANA-USD", "YFI-USD", "DNT-USD", "DOGE-USD", "ENJ-USD", "MLN-USD", "EOS-USD", "ETC-USD", "FET-USD", "FIL-USD", "ZEN-USD", "RLC-USD", "ICP1-USD", "IOTX-USD", "KNC-USD", "LRC-USD", "MKR-USD", "MIR-USD", "NKN-USD", "NU-USD", "NMR-USD", "OMG-USD", "OXT-USD", "DOT1-USD", "MATIC-USD", "QNT-USD", "SKL-USD", "SOL1-USD", "STORJ-USD", "SUSHI-USD", "SNX-USD", "SUSHI-USD", "SNX-USD", "XTZ-USD", "GRT2-USD", "TRUE-USD", "UMA-USD", "UNI3-USD", "YFI-USD", "ZEC-USD", "Predict Best (This will take a while to run!)")
# 75 Coins
# Around 6 hrs runtime to find best
selected = st.selectbox("Select Coin to Predict:", coins)
time.sleep(10)
time = st.slider("Days of Prediction:", 1, 25)
data_load_state = st.text("Progress will be Shown Here")
other_coin_ratios = st.text("")
if selected != "Predict Best (This will take a while to run!)":
    predict(selected, time)
else:
    current_values = []
    predicted_values = []
    i = 0
    coin_ratios = ""
    # Find the Predicted and Current Values
    while i < len(coins) - 1:
        selected = coins[i]
        values = predict(selected, time)
        current_values.append(values[1])
        predicted_values.append(values[0])
        coin_ratios += coins[i] + " Has a Predicted Return of " + str(float(predicted_values[i]) / float(current_values[i])) + "\n"
        other_coin_ratios.text(coin_ratios)
        i += 1
    # See Which Value the AI Predicts will bring the greatest profit
    ratios = []
    i = 0
    while i < len(current_values):
        ratios.append(float(predicted_values[i]) / float(current_values[i]))
        i += 1
    # If None Profit, use Tether
    greatest_return = max(ratios)
    if greatest_return <= 1.05:
        data_load_state.text("This Program Believes You Should Buy USDT (Tether). \nThe Profits are not Predicted to be High, or even Positive Today")
    else:
        index = ratios.index(greatest_return)
        data_load_state.text("This Program Believes You Should Buy " + str(coins[index]) + "\n The program predicts a total profit ratio of " + str(ratios[index]))
    i = 0