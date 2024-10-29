#!/usr/bin/env python
# coding: utf-8

# # Stock Price Viewer Application
# 
# This project implements a **Stock Price Viewer** application using **Dash** and **Plotly** to visualize intraday stock prices of publicly traded companies. The app fetches real-time stock data from the **Alpha Vantage** API, allowing users to explore and download stock prices based on various intervals.
# 
# ## Key Features
# 
# - **Intraday Data Fetching**: Retrieve stock data at various intervals (1, 5, 15, 30, and 60 minutes).
# - **Interactive Visualization**: Display stock prices in a dynamic graph using Plotly.
# - **User Input**: Allow users to specify the stock symbol and date range for data visualization.
# - **Data Download**: Enable users to download the displayed data as a CSV file for offline analysis.
# 
# ## Data Source
# 
# ### Alpha Vantage API
# - **Function Used**: `TIME_SERIES_INTRADAY`
# - **Parameters**:
#   - **symbol**: Stock ticker symbol (e.g., AAPL for Apple).
#   - **interval**: Time interval between data points (1min, 5min, etc.).
#   - **apikey**: Unique API key for accessing the Alpha Vantage service.
#   
# The API returns time-series data which includes:
# - **Open**: Price at market open.
# - **High**: Highest price during the interval.
# - **Low**: Lowest price during the interval.
# - **Close**: Price at market close.
# - **Volume**: Number of shares traded.

# In[3]:


import requests
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
from dash.dependencies import Input, Output
from dash import dcc
from dash import html


# In[4]:


api_key = '5ZI7ER0IIJOHIP7J'


# In[5]:


def fetch_intraday_data(symbol, interval, api_key):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={api_key}&outputsize=full"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        # Print response to check the structure
        print("API response:", data)  # For debugging, comment out after confirming
        
        # Check for errors in the response
        if 'Time Series (5min)' not in data:
            raise ValueError("The data returned by the API does not contain 'Time Series (5min)'. Check API key, symbol, or interval.")

        # Proceed if 'Time Series (5min)' exists
        df = pd.DataFrame.from_dict(data['Time Series (5min)'], orient='index')
        df = df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume"
        })
        df = df.astype(float)
        
        return df

    except ValueError as ve:
        print("ValueError:", ve)
        return pd.DataFrame()  # Return empty DataFrame on error
    except Exception as e:
        print("An error occurred:", e)
        return pd.DataFrame()


# In[6]:


# Initialize the Dash app
app = dash.Dash(__name__)

# Update layout with blue-pink theme
app.layout = html.Div(
    style={'backgroundColor': '#f0f4ff', 'fontFamily': 'Arial', 'padding': '20px'},
    children=[
        html.H1("Stock Price Viewer", style={'color': '#6c63ff'}),
        html.Div("View and download intraday stock prices.", style={'color': '#3b3b3b'}),
        dcc.Input(
            id='stock-symbol', type='text', value='AAPL', placeholder='Enter stock symbol',
            style={'margin': '10px 0', 'padding': '10px', 'border': '2px solid #6c63ff'}
        ),
        dcc.Dropdown(
            id='interval-dropdown',
            options=[
                {'label': '1 Minute', 'value': '1min'},
                {'label': '5 Minutes', 'value': '5min'},
                {'label': '15 Minutes', 'value': '15min'},
                {'label': '30 Minutes', 'value': '30min'},
                {'label': '60 Minutes', 'value': '60min'}
            ],
            value='5min',
            style={'width': '200px', 'color': '#6c63ff', 'border': '1px solid #3b3b3b'}
        ),
        dcc.DatePickerRange(
            id='date-picker',
            start_date=pd.to_datetime('today') - pd.DateOffset(days=30),
            end_date=pd.to_datetime('today'),
            style={'color': '#6c63ff'}
        ),
        html.Button("Download Data", id="btn-download", style={'backgroundColor': '#6c63ff', 'color': '#ffffff'}),
        dcc.Download(id="download-dataframe-csv"),
        dcc.Graph(id='stock-graph')
    ]
)

# Callback for updating the graph
@app.callback(
    Output('stock-graph', 'figure'),
    Input('stock-symbol', 'value'),
    Input('interval-dropdown', 'value'),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date')
)
def update_graph(symbol, interval, start_date, end_date):
    try:
        df = fetch_intraday_data(symbol, interval, api_key)
        if df.empty:
            raise ValueError("No data found for the selected date range or stock symbol.")
        df.index = pd.to_datetime(df.index)
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        # Create figure with blue-pink colors
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='#6c63ff')))
        fig.update_layout(
            title={'text': f'{symbol} Intraday Prices', 'font': {'size': 24, 'color': '#6c63ff'}},
            xaxis={'title': 'Time'},
            yaxis={'title': 'Price (USD)'},
            plot_bgcolor='#f9f9ff',
            paper_bgcolor='#f0f4ff'
        )
        return fig

    except Exception as e:
        fig = go.Figure()
        fig.update_layout(
            title="Error",
            xaxis={'title': 'Time'},
            yaxis={'title': 'Price (USD)'},
            annotations=[{
                'text': str(e),
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 20, 'color': 'red'}
            }],
            plot_bgcolor='#f9f9ff',
            paper_bgcolor='#f0f4ff'
        )
        return fig

# Callback for downloading CSV
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn-download", "n_clicks"),
    Input('stock-symbol', 'value'),
    Input('interval-dropdown', 'value'),
    prevent_initial_call=True,
)
def download_data(n_clicks, symbol, interval):
    df = fetch_intraday_data(symbol, interval, api_key)
    if df.empty:
        raise PreventUpdate
    return dict(content=df.to_csv(index=True), filename=f"{symbol}_data.csv")

if __name__ == '__main__':
    app.run_server(debug=True)


# ## For the stock symbol input, we can enter the ticker symbol of any publicly traded company. Ticker symbols are short, unique identifiers for companies on the stock market. Here are some examples:
# 
# ### AAPL for Apple Inc.
# ### GOOGL for Alphabet Inc. (Google's parent company)
# ### MSFT for Microsoft Corporation
# ### AMZN for Amazon.com, Inc.
# ### TSLA for Tesla, Inc.
# ### If you enter one of these (e.g., AAPL for Apple), My app will fetch and display the stock's intraday prices based on the selected interval.

# In[8]:


symbol = "AAPL"  # Use a stock symbol that you know exists
interval = "5min"
api_key = "YOUR_API_KEY"

# Fetch data
try:
    df = fetch_intraday_data(symbol, interval, api_key)
    print(df.head())  # Check the data output
except Exception as e:
    print(f"Error fetching data: {e}")


# # Fetching Data
# The function fetch_intraday_data(symbol, interval, api_key) is responsible for making requests to the Alpha Vantage API and returning the stock data as a DataFrame. It includes error handling to manage potential issues with the API response.
# 
# # Dash Application
# App Initialization: The Dash app is initialized and styled with a blue-pink theme.
# Layout Components:
# Input for Stock Symbol: Text input for users to enter a stock ticker.
# Dropdown for Interval Selection: Users can choose the time interval for data.
# Date Picker Range: Users can select the date range for which they want to view stock prices.
# Download Button: A button to download the displayed data as a CSV file.
# Graph Component: Displays the stock prices dynamically.
# # Callbacks
# Graph Update: The graph updates based on user input (stock symbol, interval, and date range).
# CSV Download: The app allows users to download the displayed data when the button is clicked.
# Usage Instructions
# To use the application:
# 
# # Run the application using the command app.run_server(debug=True).
# Enter the stock symbol (e.g., AAPL for Apple, GOOGL for Google).
# Select the desired time interval from the dropdown menu.
# Choose the date range using the date picker.
# Click the Download Data button to save the data as a CSV file.
# Example Stock Symbols
# AAPL: Apple Inc.
# GOOGL: Alphabet Inc. (Google's parent company)
# MSFT: Microsoft Corporation
# AMZN: Amazon.com, Inc.
# TSLA: Tesla, Inc.

# # Conclusion
# This Stock Price Viewer application provides a user-friendly interface to analyze intraday stock performance. By utilizing the Alpha Vantage API, it demonstrates how to fetch, visualize, and export financial data effectively.
