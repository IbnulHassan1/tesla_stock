import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.model_selection import train_test_split
import streamlit as st

# Streamlit app
st.title("Tesla Stock Price Prediction")
st.write("View predictions for Tesla stock prices using a GRU model.")

# Load the data directly from the file in the same folder
file_path = "TSLA.xlsx"
try:
    # Load the data
    stock_data = pd.read_excel(file_path)

    # Convert the 'Date' column to datetime
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')
    stock_data = stock_data.dropna(subset=['Date'])  # Drop rows with invalid dates

    # Filter the date range
    stock_data_filtered = stock_data[
        (stock_data['Date'] >= '2000-01-01') & (stock_data['Date'] <= '2023-12-31')
    ]

    # Select and scale the 'Close' prices
    close_prices = stock_data_filtered['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Parameter selection
    lookback = st.slider("Select Lookback Period (days)", 30, 180, 120)
    epochs = st.slider("Select Number of Training Epochs", 1, 50, 5)

    # Create the training dataset
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, 0])  # Use the past 'lookback' days
        y.append(scaled_data[i, 0])  # Predict the next day
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for GRU input

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the GRU model
    model_gru = Sequential([
        GRU(100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        GRU(50, return_sequences=False),
        Dense(25, activation='relu'),
        Dense(1)
    ])

    # Compile and train the model
    model_gru.compile(optimizer='adam', loss='mean_squared_error')
    with st.spinner("Training the GRU model..."):
        model_gru.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=1)

    # Predict the next 365 days
    last_known_data = scaled_data[-lookback:]  # Get the last 'lookback' days from the actual data
    predicted_prices = []

    for _ in range(365):  # Predict the next 365 days
        input_data = last_known_data.reshape(1, lookback, 1)  # Reshape for GRU input
        prediction = model_gru.predict(input_data)  # Make prediction
        predicted_prices.append(prediction[0])  # Add predicted price to the list
        last_known_data = np.append(last_known_data[1:], prediction)  # Slide window

    # Convert predictions back to the original scale
    predicted_prices_rescaled = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

    # Generate future dates
    last_date = stock_data_filtered['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=365)

    # Plot the results
    st.subheader("Prediction Results")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(stock_data_filtered['Date'], stock_data_filtered['Close'], label='Actual Prices', color='blue')
    ax.plot(future_dates, predicted_prices_rescaled, label='Predicted Prices (GRU)', color='orange', linestyle='--')
    ax.set_title("TESLA Stock Price Prediction for the Next Year (GRU)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)

except FileNotFoundError:
    st.error(f"File '{file_path}' not found in the current directory.")
