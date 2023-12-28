import time
import numpy as np
import yfinance as yf
import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import streamlit as st
import matplotlib.pyplot as plt


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
    else:
        st.error("Error: Empty dataset after clustering.")


if __name__ == '__main__':
    main()
