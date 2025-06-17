import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Tech Stock Performance Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    """Load and cache the processed stock data."""
    try:
        data = pd.read_csv("processed_tech_stock_data.csv", index_col=0, parse_dates=True)
        # Ensure the index is a DatetimeIndex and remove timezone if present
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        # Normalize the datetime index to remove time component
        data.index = data.index.normalize()
        return data
    except FileNotFoundError:
        st.error("Data file not found. Please run the data processing scripts first.")
        return None

# Plotting functions
def plot_stock_prices(df, selected_tickers):
    """Plot stock prices for selected tickers."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for ticker in selected_tickers:
        ax.plot(df.index, df[f'close_{ticker.lower()}'], label=ticker.upper())
    ax.set_title('Stock Prices Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()
    return fig

def plot_cumulative_returns(df, selected_tickers):
    """Plot cumulative returns for selected tickers."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for ticker in selected_tickers:
        ax.plot(df.index, df[f'cumulative_return_{ticker.lower()}'], label=ticker.upper())
    ax.set_title('Cumulative Returns')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()
    return fig

def plot_daily_returns_distribution(df, selected_tickers):
    """Plot distribution of daily returns for selected tickers."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for ticker in selected_tickers:
        sns.histplot(df[f'daily_return_{ticker.lower()}'].dropna(), kde=True, label=ticker.upper(), alpha=0.6, ax=ax)
    ax.set_title('Distribution of Daily Returns')
    ax.set_xlabel('Daily Return')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True)
    return fig

def plot_correlation_heatmap(df, selected_tickers):
    """Plot correlation heatmap of daily returns for selected tickers."""
    returns_df = pd.DataFrame()
    for ticker in selected_tickers:
        returns_df[ticker.upper()] = df[f'daily_return_{ticker.lower()}']
    
    correlation_matrix = returns_df.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
    ax.set_title('Correlation Matrix of Daily Returns')
    return fig

def plot_volatility(df, selected_tickers):
    """Plot volatility for selected tickers."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for ticker in selected_tickers:
        ax.plot(df.index, df[f'volatility_{ticker.lower()}'], label=ticker.upper())
    ax.set_title('Rolling Volatility (21-day)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility')
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()
    return fig

# Main app
def main():
    st.title("📈 Tech Stock Performance Analysis Dashboard")
    st.markdown("Interactive analysis of major technology stocks from 2020-2024")
    
    # Load data
    data = load_data()
    if data is None:
        return
    
    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    
    # Stock selection
    available_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "TSLA"]
    selected_tickers = st.sidebar.multiselect(
        "Select Stocks to Analyze:",
        available_tickers,
        default=["AAPL", "MSFT", "GOOGL"]
    )
    
    if not selected_tickers:
        st.warning("Please select at least one stock to analyze.")
        return
    
    # Date range selection
    min_date = data.index.min().date()
    max_date = data.index.max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_data = data[(data.index.date >= start_date) & (data.index.date <= end_date)]
    else:
        filtered_data = data
    
    # Analysis type selection
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type:",
        ["Stock Prices", "Cumulative Returns", "Daily Returns Distribution", "Correlation Analysis", "Volatility Analysis"]
    )
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(f"{analysis_type} - {', '.join(selected_tickers)}")
        
        # Generate and display plots based on selection
        if analysis_type == "Stock Prices":
            fig = plot_stock_prices(filtered_data, selected_tickers)
            st.pyplot(fig)
            
        elif analysis_type == "Cumulative Returns":
            fig = plot_cumulative_returns(filtered_data, selected_tickers)
            st.pyplot(fig)
            
        elif analysis_type == "Daily Returns Distribution":
            fig = plot_daily_returns_distribution(filtered_data, selected_tickers)
            st.pyplot(fig)
            
        elif analysis_type == "Correlation Analysis":
            fig = plot_correlation_heatmap(filtered_data, selected_tickers)
            st.pyplot(fig)
            
        elif analysis_type == "Volatility Analysis":
            fig = plot_volatility(filtered_data, selected_tickers)
            st.pyplot(fig)
    
    with col2:
        st.subheader("Summary Statistics")
        
        # Display summary statistics for selected stocks
        for ticker in selected_tickers:
            with st.expander(f"{ticker} Stats"):
                close_col = f'close_{ticker.lower()}'
                return_col = f'daily_return_{ticker.lower()}'
                
                if close_col in filtered_data.columns and return_col in filtered_data.columns:
                    current_price = filtered_data[close_col].iloc[-1]
                    price_change = filtered_data[close_col].iloc[-1] - filtered_data[close_col].iloc[0]
                    price_change_pct = (price_change / filtered_data[close_col].iloc[0]) * 100
                    avg_return = filtered_data[return_col].mean() * 100
                    volatility = filtered_data[return_col].std() * 100
                    
                    st.metric("Current Price", f"${current_price:.2f}")
                    st.metric("Price Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
                    st.metric("Avg Daily Return", f"{avg_return:.3f}%")
                    st.metric("Volatility", f"{volatility:.3f}%")
    
    # Additional insights
    st.subheader("Key Insights")
    
    if len(selected_tickers) > 1:
        # Calculate correlation insights
        returns_df = pd.DataFrame()
        for ticker in selected_tickers:
            returns_df[ticker] = filtered_data[f'daily_return_{ticker.lower()}']
        
        correlation_matrix = returns_df.corr()
        
        # Find highest and lowest correlations
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        correlation_matrix_masked = correlation_matrix.mask(mask)
        
        max_corr = correlation_matrix_masked.max().max()
        min_corr = correlation_matrix_masked.min().min()
        
        max_pair = correlation_matrix_masked.stack().idxmax()
        min_pair = correlation_matrix_masked.stack().idxmin()
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Highest Correlation:** {max_pair[0]} & {max_pair[1]} ({max_corr:.3f})")
        with col2:
            st.info(f"**Lowest Correlation:** {min_pair[0]} & {min_pair[1]} ({min_corr:.3f})")

if __name__ == "__main__":
    main()

