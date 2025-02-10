import yfinance as yf
import time
import pandas as pd
import os
import glob


def data_collector():
    Russel_3000 = pd.read_csv('C:\\Projects\\Repos\\Public\\Portfolio-Simulation\\Russell_3000_Tickers_20200629.csv')
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))

    gatherd_data = glob.glob(os.path.join(app_path, "Data", "*.csv"))

    for i in range(len(Russel_3000)):
        symbol = Russel_3000.iloc[i, 1]
        expected_path = os.path.join(app_path, "Data", f"{symbol}.csv")

        if not expected_path in gatherd_data:

            ticker = yf.Ticker(symbol)
            history = ticker.history(period='30y')
            history.to_csv(expected_path)

            print(f"collected {symbol} {i}")
            time.sleep(2) # Dont spam the API

if __name__ == "__main__":
    data_collector()