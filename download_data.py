import yfinance as yf
import pandas as pd

def download_stock_data(tickers, start_date, end_date):
    """Downloads historical stock data for given tickers."""
    data = yf.download(tickers, start=start_date, end=end_date)
    return data

if __name__ == "__main__":
    tech_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "TSLA"]
    start_date = "2020-01-01"
    end_date = "2024-12-31"

    print(f"Downloading stock data for {tech_tickers} from {start_date} to {end_date}...")
    stock_data = download_stock_data(tech_tickers, start_date, end_date)

    if not stock_data.empty:
        print("Data downloaded successfully. Displaying first 5 rows:")
        print(stock_data.head())
        print("\nDisplaying info:")
        stock_data.info()
        stock_data.to_csv("tech_stock_data.csv")
        print("\nStock data saved to tech_stock_data.csv")
    else:
        print("Failed to download data.")


