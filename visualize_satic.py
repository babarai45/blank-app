import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates

def plot_individual_stock_performance(df, ticker):
    """Plots the close price and volume for a single stock."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'{ticker.upper()} Stock Performance', fontsize=16)

    # Plot Close Price
    axes[0].plot(df.index, df[f'close_{ticker.lower()}'])
    axes[0].set_title('Close Price')
    axes[0].set_ylabel('Price (USD)')
    axes[0].grid(True)

    # Plot Volume
    axes[1].bar(df.index, df[f'volume_{ticker.lower()}'], color='skyblue')
    axes[1].set_title('Volume Traded')
    axes[1].set_ylabel('Volume')
    axes[1].set_xlabel('Date')
    axes[1].grid(True)

    # Format x-axis as dates
    fig.autofmt_xdate()

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(f'static_plots/{ticker.lower()}_price_volume.png')
    plt.close()

def plot_cumulative_returns(df, tickers):
    """Plots cumulative returns for selected stocks."""
    plt.figure(figsize=(14, 7))
    for ticker in tickers:
        plt.plot(df.index, df[f'cumulative_return_{ticker.lower()}'], label=ticker.upper())
    plt.title('Cumulative Returns of Tech Stocks')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig('static_plots/cumulative_returns.png')
    plt.close()

def plot_daily_returns_distribution(df, tickers):
    """Plots the distribution of daily returns for selected stocks."""
    plt.figure(figsize=(14, 7))
    for ticker in tickers:
        sns.histplot(df[f'daily_return_{ticker.lower()}'].dropna(), kde=True, label=ticker.upper(), alpha=0.6)
    plt.title('Distribution of Daily Returns')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('static_plots/daily_returns_distribution.png')
    plt.close()

def plot_correlation_heatmap(df, tickers):
    """Plots a heatmap of daily returns correlation."""
    returns_df = pd.DataFrame()
    for ticker in tickers:
        returns_df[ticker.upper()] = df[f'daily_return_{ticker.lower()}']

    correlation_matrix = returns_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Daily Returns')
    plt.tight_layout()
    plt.savefig('static_plots/correlation_heatmap.png')
    plt.close()

if __name__ == "__main__":
    try:
        processed_data = pd.read_csv("processed_tech_stock_data.csv", index_col=0, parse_dates=True)
        print("Processed data loaded successfully. Shape:", processed_data.shape)

        # Ensure the index is a DatetimeIndex and remove timezone if present
        if processed_data.index.tz is not None:
            processed_data.index = processed_data.index.tz_localize(None)
        # Normalize the datetime index to remove time component
        processed_data.index = processed_data.index.normalize()

        tech_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "TSLA"]

        import os
        if not os.path.exists("static_plots"):
            os.makedirs("static_plots")

        print("Generating individual stock performance plots...")
        for ticker in tech_tickers:
            plot_individual_stock_performance(processed_data.copy(), ticker)

        print("Generating cumulative returns plot...")
        plot_cumulative_returns(processed_data.copy(), tech_tickers)

        print("Generating daily returns distribution plot...")
        plot_daily_returns_distribution(processed_data.copy(), tech_tickers)

        print("Generating correlation heatmap...")
        plot_correlation_heatmap(processed_data.copy(), tech_tickers)

        print("All static plots generated and saved in the 'static_plots' directory.")

    except FileNotFoundError:
        print("Error: processed_tech_stock_data.csv not found. Please run preprocess_data.py first.")
    except Exception as e:
        print(f"An error occurred: {e}")


