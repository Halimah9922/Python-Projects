# Stock Price Viewer Application

## Overview

The **Stock Price Viewer** application allows users to visualize intraday stock prices of publicly traded companies using **Dash** and **Plotly**. The app fetches real-time stock data from the **Alpha Vantage** API, enabling users to explore and download stock prices based on various intervals.

## Key Features

- **Intraday Data Fetching**: Retrieve stock data at various intervals (1, 5, 15, 30, and 60 minutes).
- **Interactive Visualization**: Display stock prices in a dynamic graph using Plotly.
- **User Input**: Allow users to specify the stock symbol and date range for data visualization.
- **Data Download**: Enable users to download the displayed data as a CSV file for offline analysis.

## Data Source

### Alpha Vantage API

- **Function Used**: `TIME_SERIES_INTRADAY`
- **Parameters**:
  - **symbol**: Stock ticker symbol (e.g., AAPL for Apple).
  - **interval**: Time interval between data points (1min, 5min, etc.).
  - **apikey**: Unique API key for accessing the Alpha Vantage service.

The API returns time-series data which includes:
- **Open**: Price at market open.
- **High**: Highest price during the interval.
- **Low**: Lowest price during the interval.
- **Close**: Price at market close.
- **Volume**: Number of shares traded.

## Installation

To run the Stock Price Viewer application, ensure you have the following dependencies installed:

```bash
pip install dash plotly requests pandas
