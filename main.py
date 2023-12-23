import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
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

    # Handle missing values by replacing with the mean
    stock_data = stock_data.fillna(stock_data.mean())

    # Print column names and levels for inspection
    st.write("Column Names:", stock_data.columns.tolist())
    st.write("Column Levels:", stock_data.columns.levels)

    st.write("Full DataFrame:", stock_data)

    return stock_data.pct_change().dropna()


# Function to perform clustering
def perform_clustering(data, num_clusters):
    st.write("Data Size Before Clustering:", data.shape)

    # Check for missing values
    st.write("Missing Values Before Imputation:")
    st.write(data.isnull().sum())

    # Check for infinite or NaN values
    st.write("Infinite/NaN Values Before Handling:")
    st.write(data.isin([np.inf, -np.inf, np.nan]).sum())

    # Replace infinite values with NaN and handle missing values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data)

    # Check for missing values after imputation
    st.write("Missing Values After Imputation:")
    st.write(pd.DataFrame(data_imputed).isnull().sum())

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_imputed)

    # Apply PCA for dimensionality reduction
    n_components = min(scaled_data.shape[0], scaled_data.shape[1]) - 1
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(scaled_data)

    # Check for infinite or NaN values after preprocessing
    st.write("Infinite/NaN Values After Preprocessing:")
    st.write(np.isinf(reduced_data).sum())

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(reduced_data)

    return clusters


# Function to calculate correlation and present top correlations
def calculate_and_present_correlation(selected_stocks_analysis):
    # Concatenate the selected stocks into a single DataFrame
    stock_data_concatenated = pd.concat(selected_stocks_analysis, axis=1)

    # Calculate correlation matrix
    correlation_matrix = stock_data_concatenated.corr()

    # Extract top 10 positive and negative correlations
    top_positive_corr = correlation_matrix.unstack().sort_values(ascending=False).drop_duplicates().head(10)
    top_negative_corr = correlation_matrix.unstack().sort_values().drop_duplicates().head(10)

    return top_positive_corr, top_negative_corr


# Function to perform EDA for each selected stock
def perform_eda(stock_data, stock_name):
    st.write(f"Exploratory Data Analysis for {stock_name}")

    # Get the available columns for the selected stock
    available_columns = stock_data.columns.get_level_values(0).unique().tolist()

    # Select the column for analysis
    selected_column = st.selectbox("Select a column for analysis:", available_columns)

    # Check if the selected column is present in the dataset
    if selected_column in available_columns:
        # Extract the data for the selected column
        column_data = stock_data[selected_column][stock_name]

        # Flatten the MultiIndex columns to simplify indexing
        column_data.columns = [f'{col[0]}_{col[1]}' for col in column_data.columns]

        # Visualize temporal structure
        st.write("Temporal Structure:")
        temporal_chart = alt.Chart(column_data.reset_index()).mark_line().encode(
            x='index:T',
            y=f'{selected_column}_{stock_name}:Q'
        ).properties(width=600, height=300)
        st.altair_chart(temporal_chart)

        # Weekly Changes
        st.write("Changes in Distribution Over Intervals:")
        weekly_changes = column_data.groupby(column_data.index.dayofweek).mean()
        st.bar_chart(weekly_changes)

        # Distribution Analysis
        st.write("Distribution Analysis:")
        fig, ax = plt.subplots()
        ax.hist(column_data.stack(), bins=30, alpha=0.7)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    else:
        st.error(f"Column '{selected_column}' not found in the dataset.")


# Main Streamlit app
def main():
    st.title("Stock Grouping, Correlation, and EDA Analysis")

    # Fetch historical stock data for NASDAQ-100 companies
    stock_data = get_nasdaq_100_data()

    # Display data size before clustering
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

            # Initialize selected_stock_data to None
            selected_stock_data = None

            # Ensure that there are stocks in the cluster before selecting
            if stocks_indices_in_cluster:
                # Use the first stock in the cluster for analysis
                selected_stock_index = np.random.choice(stocks_indices_in_cluster)  # Randomly select one stock
                selected_stock_name = stock_data.columns[selected_stock_index]  # Define selected_stock_name here

                try:
                    # Use iloc to select by index
                    selected_stock_data = stock_data.iloc[:, selected_stock_index:selected_stock_index + 1]
                    selected_stock_data.columns = [selected_stock_name]  # Set the column name to the stock's name
                    selected_stock_data.reset_index(inplace=True, drop=True)  # Reset the index to integer index
                    selected_stocks_analysis.append(selected_stock_data)

                    # Display the selected stocks for analysis
                    st.write(f"Selected Stock for Group {i + 1}:", selected_stock_name)

                except IndexError:
                    st.warning(f"No stock found in group {i + 1}.")
            else:
                st.warning(f"No stocks found in group {i + 1}.")

        # Perform EDA for each selected stock
        for selected_stock_data in selected_stocks_analysis:
            selected_stock_name = str(selected_stock_data.columns[0])  # Convert to string
            st.write(f"Exploratory Data Analysis for {selected_stock_name}")

            # Call the perform_eda function with the selected stock data and name
            perform_eda(selected_stock_data, selected_stock_name)

        # Calculate and present correlation
        top_positive_corr, top_negative_corr = calculate_and_present_correlation(selected_stocks_analysis)

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
