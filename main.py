import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


# Function to fetch historical stock data for NASDAQ-100 companies
def get_nasdaq_100_data():
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
    return stock_data['Adj Close'].pct_change().dropna()


# Function to perform clustering
def perform_clustering(data, num_clusters):
    if data.shape[0] == 0 or data.shape[1] == 0:
        st.error("Error: Empty dataset before clustering.")
        return []

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Apply PCA for dimensionality reduction
    n_components = min(scaled_data.shape[0],
                       scaled_data.shape[1]) - 1  # Set n_components to the minimum of samples and features
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(scaled_data)

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
def perform_eda(stock_data, stock_name):
    st.write(f"Exploratory Data Analysis for {stock_name}")

    # Visualize temporal structure
    st.write("Temporal Structure:")
    st.line_chart(stock_data)

    # Visualize distribution with a histogram
    st.write("Distribution Analysis:")
    fig, ax = plt.subplots()
    ax.hist(stock_data, bins=30, alpha=0.7)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Changes in Distribution Over Intervals
    st.write("Changes in Distribution Over Intervals:")

    # Check if data is a DataFrame before trying to access columns
    if isinstance(stock_data, pd.DataFrame):
        # Check available column names
        st.write("Available Columns:", stock_data.columns.tolist())

        # Replace 'Adj Close' with the correct column name
        close_column_name = 'Adj Close'  # Replace with the correct column name
        if close_column_name in stock_data.columns:
            weekly_changes = stock_data.groupby(stock_data.index.weekday)[close_column_name].mean()
            st.bar_chart(weekly_changes)
        else:
            st.error(f"Column '{close_column_name}' not found in the dataset.")
    else:
        st.warning("Data is not a DataFrame.")


# Main Streamlit app
def main():
    st.title("Stock Grouping, Correlation, and EDA Analysis")

    # Fetch historical stock data for NASDAQ-100 companies
    stock_data = get_nasdaq_100_data()

    # Display data size before clustering1
    st.write("Data Size Before Clustering:", stock_data.shape)

    # Perform clustering with k-means (number of clusters = 4)
    num_clusters = 4
    clusters = perform_clustering(stock_data, num_clusters)

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

        # Perform EDA for each selected stock
        for selected_stock_name in selected_stocks_analysis:
            selected_stock_data = stock_data[[selected_stock_name]]  # Ensure the correct data type (DataFrame)
            if isinstance(selected_stock_data, pd.DataFrame):
                perform_eda(selected_stock_data, selected_stock_name)

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
