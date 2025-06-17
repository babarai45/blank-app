import pandas as pd

def preprocess_stock_data(df):
    """Preprocesses the stock data: cleans, calculates returns, moving averages, and volatility."""
    # Flatten multi-level columns
    df.columns = [f"{col[0].lower()}_{col[1].lower()}" for col in df.columns]

    # Rename 'adj close' to 'close' for simplicity if it exists
    df.rename(columns={col: col.replace('adj close', 'close') for col in df.columns if 'adj close' in col}, inplace=True)

    # Calculate daily returns and cumulative returns for each ticker
    tickers = sorted(list(set([col.split('_')[1] for col in df.columns if '_' in col])))
    for ticker in tickers:
        close_col = f"close_{ticker.lower()}"
        if close_col in df.columns:
            df[f"daily_return_{ticker.lower()}"] = df[close_col].pct_change()
            df[f"cumulative_return_{ticker.lower()}"] = (1 + df[f"daily_return_{ticker.lower()}"]).cumprod() - 1

            # Calculate moving averages
            df[f"ma50_{ticker.lower()}"] = df[close_col].rolling(window=50).mean()
            df[f"ma200_{ticker.lower()}"] = df[close_col].rolling(window=200).mean()

            # Calculate rolling volatility (21-day standard deviation of daily returns)
            df[f"volatility_{ticker.lower()}"] = df[f"daily_return_{ticker.lower()}"].rolling(window=21).std()

    return df

if __name__ == "__main__":
    try:
        stock_data = pd.read_csv("tech_stock_data.csv", header=[0, 1], index_col=0, parse_dates=True)
        print("Original data loaded successfully. Shape:", stock_data.shape)
        print("Original columns:", stock_data.columns)

        processed_data = preprocess_stock_data(stock_data.copy())

        if not processed_data.empty:
            print("\nProcessed data info:")
            processed_data.info()
            processed_data.to_csv("processed_tech_stock_data.csv")
            print("\nProcessed stock data saved to processed_tech_stock_data.csv")
        else:
            print("Failed to preprocess data.")

    except FileNotFoundError:
        print("Error: tech_stock_data.csv not found. Please run download_data.py first.")
    except Exception as e:
        print(f"An error occurred: {e}")


