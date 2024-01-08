import time
import numpy as np
import yfinance as yf
import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Define the columns used for LSTM modeling
LSTM_Selected_Columns = ['Open', 'High', 'Low', 'Close']


# Function to fetch historical stock data for NASDAQ-100 companies
def get_nasdaq_100_data(selected_column='Adj Close'):
    nasdaq_100_symbols = ["AAPL", "MSFT", "AMZN", "AVGO", "META",
                          "TSLA", "NVDA", "GOOGL", "GOOG", "COST",
                          "ADBE", "PEP", "AMD", "NFLX", "CSCO",
                          "INTC", "TMUS", "CMCSA", "INTU", "QCOM",
                          "TXN", "AMGN", "HON", "AMAT", "BKNG",
                          "ISRG", "SBUX", "VRTX", "LRCX", "GILD",
                          "ADI", "MDLZ", "PDD", "MU", "ADP",
                          "PANW", "REGN", "MELI", "SNPS", "KLAC",
                          "CDNS", "CSX", "PYPL", "ASML", "MAR",
                          "LULU", "ABNB", "CTAS", "NXPI", "MNST",
                          "CRWD", "ROP", "WDAY", "CHTR", "ORLY",
                          "MRVL", "ADSK", "PCAR", "MCHP", "DXCM",
                          "CPRT", "ROST", "IDXX", "KDP", "FTNT",
                          "ODFL", "KHC", "PAYX", "AEP", "AZN",
                          "CTSH", "BIIB", "FAST", "TEAM", "DASH",
                          "EA", "DDOG", "CEG", "ON", "MRNA",
                          "CSGP", "GEHC", "EXC", "BKR", "VRSK",
                          "XEL", "GFS", "ZS", "TTD", "ANSS",
                          "CDW", "CCEP", "DLTR", "MDB", "FANG",
                          "TTWO", "WBD", "SPLK", "WBA", "ILMN",
                          "SIRI"
                          ]

    stock_data = yf.download(nasdaq_100_symbols, start="2022-12-01", end="2023-12-01")

    st.write("Downloaded Data:", stock_data)
    st.write("Column Names:", stock_data.columns)

    # Apply log transformation to the 'Volume' column if it exists in the DataFrame
    if ('Volume', selected_column) in stock_data.columns:
        volume_column = ('Volume', selected_column)

        # Identify non-zero values
        non_zero_mask = stock_data[volume_column] > 0

        # Apply log transformation only to non-zero values
        stock_data[volume_column] = np.log1p(stock_data[volume_column])

        # Check if there are any non-zero values after log transformation
        if non_zero_mask.any():
            stock_data = stock_data[non_zero_mask]  # Filter out rows with zero values after log transformation
        else:
            st.warning(f"No non-zero values found in the log-transformed 'Volume' column after removing outliers.")
            # Handle the case when there are no non-zero values

    # Check if the DataFrame is empty after preprocessing
    if stock_data.empty:
        st.warning("Empty DataFrame after preprocessing. Check your data and preprocessing steps.")

    # Check if the selected column is in the available columns
    if selected_column in stock_data.columns.get_level_values(0):
        st.write("DataFrame Info:", stock_data.info())
        return stock_data[selected_column].pct_change().dropna()
    else:
        raise ValueError(
            f"""Invalid column selected: {selected_column}. 
            Available columns are {stock_data.columns.get_level_values(0)}""")

    # return stock_data['High'].pct_change().dropna()


# Function to perform clustering
def perform_clustering(data, num_clusters, selected_column):
    # Check the dataset isn't empty
    if data.shape[0] == 0 or data.shape[1] == 0:
        st.error("Error: Empty dataset before clustering.")
        return []

    st.write("Column Names:", data.columns)
    st.write("DataFrame Info:", data.info())

    # Forward filling missing values
    stock_data = data.ffill()

    # Print the length of the dataset before outlier removal
    st.write(f"Length of {selected_column} in StockDataFrame before removing outliers:", len(stock_data))

    # Check if the DataFrame is empty after preprocessing
    if stock_data.empty:
        st.warning("Empty DataFrame after preprocessing. Check your data and preprocessing steps.")
        return []  # Return an empty list to signal an issue

    # Print the dataset before outlier removal
    st.write(f"{selected_column} in StockDataFrame before removing outlier:", stock_data)
    # Removing outliers using z-score
    z_scores = zscore(stock_data)  # Calculate Z-scores for each data point in the stock_data
    abs_z_scores = np.abs(z_scores)  # Take the absolute value of Z-scores
    filtered_entries = (abs_z_scores < 4).all(axis=1)  # Create a boolean mask indicating whether each data point is
    # within 3 standard deviations
    stock_data = stock_data[filtered_entries]  # Use the boolean mask to filter out rows with outliers from stock_data

    # Print the length of the dataset after outlier removal
    st.write(f"Length of {selected_column} in StockDataFrame after removing outliers:", len(stock_data))

    st.write(f"{selected_column} in StockDataFrame after removing outlier:", stock_data)

    # Normalize the data using Min-Max scaling
    scaler = MinMaxScaler()  # Create the MinMaxScaler object
    stock_data_normalized = scaler.fit_transform(stock_data)  # Fit and transform the data

    # Standardize the data using Z-Score (not in use as LSTM model will prefer normalization across different horizons)
    # scaler = StandardScaler()
    # scaled_data = scaler.fit_transform(data)

    # Apply PCA for dimensionality reduction
    n_components = min(stock_data_normalized.shape[0],
                       stock_data_normalized.shape[1]) - 1  # Set n_components to the minimum of samples and features
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(stock_data_normalized)

    st.write("Reduced StockDataFrame:", reduced_data)

    # Check if any element in reduced_data is zero
    if (reduced_data == 0).any():
        st.error("Error: Empty dataset after clustering.")
        return []

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(reduced_data)

    return clusters


# Function to calculate correlation and present top correlations
def calculate_and_present_correlation(stock_data):
    # Calculate correlation matrix
    correlation_matrix = stock_data.corr()

    # Extract top 10 positive and negative correlations
    top_positive_corr = correlation_matrix.unstack().sort_values(ascending=False).drop_duplicates().head(10)
    top_negative_corr = correlation_matrix.unstack().sort_values().drop_duplicates().head(10)

    return top_positive_corr, top_negative_corr


# Function to perform EDA for each selected stock
def perform_eda(stock_data, selected_stock_name, selected_column):
    st.write(f"Exploratory Data Analysis for {selected_stock_name} using {selected_column}")

    # Visualize temporal structure
    st.write("Temporal Structure:")
    st.line_chart(stock_data[selected_stock_name])  # Use the selected_stock_name for the line chart

    # Visualize distribution with a histogram
    st.write("Distribution Analysis:")
    fig, ax = plt.subplots()
    ax.hist(stock_data[selected_stock_name], bins=30, alpha=0.7)  # Use the selected_stock_name for the histogram
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Changes in Distribution Over Intervals
    st.write(f"Changes in Distribution Over Intervals:")
    weekly_changes = stock_data[selected_stock_name].groupby(stock_data.index.weekday).mean()
    st.bar_chart(weekly_changes)


# Function to build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(35, input_shape=input_shape, activation='relu'))
    # model.add(Dropout(dropout_rate))  # Add a dropout layer with the specified dropout rate
    model.add(Dense(1))  # Output layer with the number of features
    model.compile(optimizer='adam', loss='mse')  # Using mean squared error for regression
    return model


# Function to create sequences for LSTM
def create_sequences(df, sequence_length):
    sequences, next_day_close = [], []
    for i in range(len(df) - sequence_length):
        seq = df.iloc[i:i + sequence_length].values
        label = df.iloc[i + sequence_length].values
        sequences.append(seq)
        next_day_close.append(label)
    return np.array(sequences), np.array(next_day_close)


# Function to evaluate the LSTM model
def evaluate_lstm_model(model, x_test, y_test, scaler, df_model):
    st.write("Evaluating LSTM Model...")

    # Make predictions using the test data
    y_pred = model.predict(x_test)

    # Reshape arrays for inverse transformation
    y_pred_reshaped = y_pred.reshape(-1, 1)
    y_test_reshaped = y_test.reshape(-1, 1)

    # Invert the scaling for the predictions and actual values
    y_pred_inv = scaler.inverse_transform(np.hstack((np.zeros((y_pred_reshaped.shape[0], df_model.shape[1] - 1)),
                                                     y_pred_reshaped)))
    y_test_inv = scaler.inverse_transform(np.hstack((np.zeros((y_test_reshaped.shape[0], df_model.shape[1] - 1)),
                                                     y_test_reshaped)))

    # Extract only the 'Close' column for comparison
    y_pred_inv_close = y_pred_inv[:, -1:]
    y_test_inv_close = y_test_inv[:, -1:]

    # Ensure both arrays have the same shape
    min_length = min(len(y_test_inv_close), len(y_pred_inv_close))
    y_test_inv_close = y_test_inv_close[:min_length]
    y_pred_inv_close = y_pred_inv_close[:min_length]

    # Flatten the arrays
    y_pred_inv_close = y_pred_inv_close.flatten()
    y_test_inv_close = y_test_inv_close.flatten()

    # Calculate the range of actual close prices
    actual_close_min = np.min(y_test_inv_close)
    actual_close_max = np.max(y_test_inv_close)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test_inv_close, y_pred_inv_close)
    st.write(f"Mean Squared Error (MSE): {mse}")

    # Visualize the predictions and actual values
    st.write("Visualizing Predictions and Actual Values...")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_test_inv_close, label='Actual Close Price', linestyle='--', color='blue')
    ax.plot(y_pred_inv_close, label='Predicted Close Price', linestyle='-', color='orange')
    ax.set_xlabel("Time")
    ax.set_ylabel("Close Price")
    ax.legend()
    st.pyplot(fig)

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # Print the results
    st.write(f"Actual Close Price Range: {actual_close_min} to {actual_close_max}")
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")

    # Compare RMSE to the range
    if rmse < (actual_close_max - actual_close_min):
        st.write("RMSE is relatively small compared to the range of actual close prices.")
    else:
        st.write("RMSE is relatively large compared to the range of actual close prices.")


# Function to generate signals based on the LSTM forecast
def generate_signals(model, scaler, df_model, sequence_length, user_selected_stock_symbols):
    # Use user_selected_stock_symbols as column names
    selected_stock_data = df_model[user_selected_stock_symbols]

    # Normalize the selected stock data for LSTM
    selected_stock_data_normalized = scaler.transform(selected_stock_data)

    # Reshape the data to have three dimensions (assuming len(df_model.columns) is the sequence length)
    selected_stock_data_normalized = selected_stock_data_normalized.reshape(1, -1, len(df_model.columns))

    # Create sequences for LSTM
    x_selected, _ = create_sequences(pd.DataFrame(selected_stock_data_normalized[0], columns=df_model.columns),
                                     sequence_length)

    # Make predictions using the trained LSTM model
    predicted_close_prices = model.predict(x_selected)

    # Invert the scaling for the predictions
    predicted_close_prices_inv = scaler.inverse_transform(predicted_close_prices)

    # Extract the closing prices for each selected stock
    predicted_close_prices_inv_close = predicted_close_prices_inv[:, :len(user_selected_stock_symbols)]

    # Flatten the array
    predicted_close_prices_inv_close = predicted_close_prices_inv_close.flatten()

    # Generate signals based on the forecast
    signals = pd.Series(index=selected_stock_data.index, dtype=int)

    # Loop through each selected stock
    for i, stock_symbol in enumerate(user_selected_stock_symbols):
        signals[predicted_close_prices_inv_close[i::len(user_selected_stock_symbols)] > selected_stock_data[stock_symbol]] = 1  # Buy Signal
        signals[predicted_close_prices_inv_close[i::len(user_selected_stock_symbols)] < selected_stock_data[stock_symbol]] = -1  # Sell Signal

    return selected_stock_data, predicted_close_prices_inv_close, signals


# Main Streamlit app
def main():
    st.title("Stock Grouping, Correlation, and EDA Analysis")

    progress_text = "Operation in progress. Please wait."
    progress_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        progress_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    progress_bar.empty()

    # Create a dropdown to select the column
    selected_column = st.selectbox("Select Column", ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'])

    # Fetch historical stock data for NASDAQ-100 companies with the selected column
    stock_data = get_nasdaq_100_data(selected_column)

    # Display data size before clustering
    st.write("Data Size Before Clustering:", stock_data.shape)

    # Dynamically exclude columns containing "Volume"
    volume_columns = [col for col in stock_data.columns if 'Volume' in col]
    selected_columns = [col for col in stock_data.columns if col not in volume_columns]

    # Ensure there are selected columns after excluding "Volume"
    if not selected_columns:
        st.error("Error: No valid columns selected after excluding 'Volume'.")
        return

    # Choose relevant columns for modeling
    df_model = stock_data[selected_columns]

    # selected_stock_data = None  # Initialize selected_stock_data outside the loop

    # Perform clustering with k-means (number of clusters = 4)
    num_clusters = 4
    clusters = perform_clustering(stock_data, num_clusters, selected_column)

    # Check if clusters is not None and not an empty list
    if clusters is not None and len(clusters) > 0:
        # Display clustered groups
        st.write("Stocks Clustered into Groups:", clusters)

        # Select one stock per cluster for analysis
        selected_stocks_analysis = []
        for i in range(num_clusters):
            # Get the indices of stocks in the current cluster
            stocks_indices_in_cluster = [j for j in range(len(clusters)) if clusters[j] == i]

            # Ensure that there are stocks in the cluster before selecting
            if stocks_indices_in_cluster:
                # Use the first stock in the cluster for analysis
                selected_stock_index = stocks_indices_in_cluster[0]
                selected_stock = stock_data.columns[selected_stock_index]
                selected_stocks_analysis.append(selected_stock)

        # Display the selected stocks for analysis
        st.write("Selected Stocks for Analysis:", selected_stocks_analysis)
        st.write("Full Selected StockDataFrame:", stock_data)

        # Assign selected_stock_data outside the loop
        selected_stock_data = stock_data[selected_stocks_analysis]

        # Perform EDA for each selected stock
        for selected_stock_name in selected_stocks_analysis:
            selected_stock_data = stock_data[selected_stocks_analysis]  # Filter to include only selected stocks
            if isinstance(selected_stock_data, pd.DataFrame):
                perform_eda(selected_stock_data, selected_stock_name, selected_column)
            else:
                st.warning(f"Data for {selected_stock_name} is not a DataFrame. Cannot perform EDA.")

        # Calculate and present correlation
        top_positive_corr, top_negative_corr = calculate_and_present_correlation(
            stock_data[selected_stocks_analysis])

        # Display top positive correlations
        st.write("Top 10 Positive Correlations:")
        st.write(top_positive_corr)

        # Display top negative correlations
        st.write("Top 10 Negative Correlations:")
        st.write(top_negative_corr)

        # LSTM Model Training
        st.write("Training LSTM Model...")

        # Normalize the data for LSTM
        scaler = MinMaxScaler()
        df_normalized = pd.DataFrame(scaler.fit_transform(df_model), columns=df_model.columns)

        # Convert the dataframe into sequences for LSTM
        sequence_length = 10  # Experiment to find best result
        x, y = create_sequences(df_normalized, sequence_length)

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

        # Build and train the LSTM model
        epochs = 10  # Experiment to find best result
        batch_size = 48  # Experiment to find best result
        model = build_lstm_model(input_shape=(x.shape[1], x.shape[2]))

        # Progress bar for LSTM model training
        progress_text_lstm = "Training LSTM Model in progress. Please wait."
        progress_bar_lstm = st.progress(0, text=progress_text_lstm)

        for epoch in range(epochs):
            # Train the model for one epoch
            model.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=0)

            # Update the progress bar
            progress_bar_lstm.progress((epoch + 1) / epochs, text=progress_text_lstm)

        progress_bar_lstm.empty()  # Clear the progress bar

        # Evaluate the LSTM model
        evaluate_lstm_model(model, x_test, y_test, scaler,
                            df_model)  # Pass the 'scaler' and 'df_model' objects as arguments

        # Main content
        st.write("### Generate Trading Signals:")

        # Create a dropdown to select the stock symbol for signal generation
        user_selected_stock_symbol = st.selectbox("Select Stock Symbol for Signal Generation", selected_stocks_analysis)

        # Button to trigger signal generation
        if st.button("Generate Signals"):
            # Generate signals for the selected stock
            selected_stock_data, predicted_close_prices_inv_close, signals = generate_signals(
                model, scaler, df_model, sequence_length, user_selected_stock_symbol)

            # Display the predicted close prices
            st.write("Predicted Close Prices:")
            st.line_chart(pd.Series(predicted_close_prices_inv_close, index=selected_stock_data.index))

            # Display the generated signals
            st.write("Generated Signals:")
            st.table(signals.rename('Signal'))

            # Visualize Buy and Sell signals
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(selected_stock_data['Close'], label='Close Price', linestyle='-', color='blue')
            ax.plot(selected_stock_data.index, selected_stock_data['Close'][signals == 1], '^', markersize=10,
                    color='g', label='Buy Signal')
            ax.plot(selected_stock_data.index, selected_stock_data['Close'][signals == -1], 'v', markersize=10,
                    color='r', label='Sell Signal')
            ax.set_xlabel("Time")
            ax.set_ylabel("Close Price")
            ax.legend()
            st.pyplot(fig)

    else:
        st.error("Error: Empty dataset after clustering.")


if __name__ == '__main__':
    main()
