# import libraries
import pandas as pd
import yfinance as yf
import datetime
from datetime import date, timedelta
import streamlit as st # deployment
import plotly.graph_objects as go # Candlestick chart
from sklearn.linear_model import LinearRegression # Time series analysis
from sklearn.preprocessing import PolynomialFeatures # Polynomial Regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Import warnings + watermark
from watermark import watermark
from warnings import filterwarnings
filterwarnings("ignore")
print(watermark())
print(watermark(iversions=True, globals_=globals()))


#------------------------------------------------------------------#

# Set page configurations - ALWAYS at the top
st.set_page_config(page_title="Crypto Predictor",page_icon="📈",layout="centered",initial_sidebar_state="auto")

@st.cache_data # Add cache data decorator

# Load and Use local style.css file
def local_css(file_name):
    """
    Use a local style.css file.
    """
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# load css file
local_css("./style/style.css")


#------------------------------------------------------------------#

# Read ticker symbols from a CSV file
try:
    tickers = pd.read_csv("./Resources/tickers.csv")
except:
    logging.error('Cannot find the CSV file')

# Title/ header
st.header("Crypto coin next day price predictor")
st.write("Select from the Top 10 crypto coins based on market cap to view forecast for tomorrow's potential closing price and compare against different Machine Learning models")
st.write("---")
    
# Show tickers list
st.write(f"<b>Below is the list of the coins available for analysis</b>",unsafe_allow_html=True)
st.write(tickers)
st.write("---")

#------------------------------------------------------------------#

# declare variable for current date/ end date
today = date.today()
end_date = today.strftime("%Y-%m-%d")

# declare variable for start date - past 3 years
d2 = date.today() - timedelta(days=1095)
start_date = d2.strftime("%Y-%m-%d")


# Display a selectbox for the user to choose a ticker
ticker = st.selectbox("Select a ticker from the dropdown menu",tickers)


# download yfinance data
data = yf.download(ticker, 
                      start=start_date, 
                      end=end_date, 
                      progress=False)
data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
data.reset_index(drop=True, inplace=True)


# Display Candlestick chart
data_check_box=st.checkbox(label=f"Display {ticker} raw dataset for the past 3 years")
if data_check_box:

    # Display full dataset
    st.write(data)

    # shape of the data
    # st.write(f"Data shape (rows, columns) - ",data.shape)



# Create Candlestick attribute
figure = go.Figure(data=[go.Candlestick(x=data["Date"],
                                        open=data["Open"], 
                                        high=data["High"],
                                        low=data["Low"], 
                                        close=data["Close"])])
figure.update_layout(title = f"Candlestick Analysis for {ticker} price",
                     xaxis_rangeslider_visible=False)


# Display Candlestick chart
candlestick_check_box=st.checkbox(label=f"Display {ticker} interactive Candlestick chart")
if candlestick_check_box:
    st.info(f"Candlestick chart shows {ticker} crypto coin's Open, High, Low, and Close price for the day")
    st.plotly_chart(figure)


st.write("---")

    
# Display data for prediction "Close"
prediction_check_box=st.checkbox(label=f"Display ML data for {ticker} next-day 'Close' price prediction")
if prediction_check_box:
    
    correlation=data.corr()
    # st.write(correlation["Close"].sort_values(ascending=False))
    
    # Create new feature - percent change
    data['Pct_Change'] = data['Close'].pct_change()
    data = data.dropna()

    # Linear Regression
    x = data[['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Pct_Change']]
    y = data[['Close']]

    lm = LinearRegression()
    lm.fit(x, y)

    # Predict tomorrow close price
    last_row = data.tail(1)
    next_day_data = pd.DataFrame({'Open': last_row['Open'],
                                  'High': last_row['High'],
                                  'Low': last_row['Low'],
                                  'Adj Close': last_row['Adj Close'],
                                  'Volume': last_row['Volume'],
                                  'Pct_Change': last_row['Pct_Change']})

    predicted_close_price = lm.predict(next_day_data)[0][0] 

    
    
    # Display the predicted close price - Linear Regression
    if st.button("Linear Regression ML prediction"):
        st.write(f"Predicted {ticker} close price for tomorrow is: <b>{predicted_close_price:.2f} USD</b>",unsafe_allow_html=True)
        
        
        
        
    
    # Display the predicted close price - Polynomial Regression
    
    # Select the 'Close' column as the target variable
    y = data['Close']

    # Select the remaining columns as the features
    X = data.drop(['Close', 'Date'], axis=1)

    # Create polynomial features with degree 2
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    # Split data into training and testing sets
    split = int(0.8*len(data))
    X_train = X_poly[:split]
    y_train = y[:split]
    X_test = X_poly[split:]
    y_test = y[split:]

    # Train a polynomial regression model
    lm_poly = LinearRegression()
    lm_poly.fit(X_train, y_train)

    # Predict tomorrow close price
    last_row = data.tail(1)
    next_day_data = pd.DataFrame({'Open': last_row['Open'],
                                  'High': last_row['High'],
                                  'Low': last_row['Low'],
                                  'Adj Close': last_row['Adj Close'],
                                  'Volume': last_row['Volume'],
                                  'Pct_Change': last_row['Pct_Change']})
    next_day_data_poly = poly.transform(next_day_data)

    predicted_close_price = lm_poly.predict(next_day_data_poly)[0]

    # Define a button to display the predicted price
    if st.button("Polynomial Regression ML prediction"):
        st.write(f"Predicted {ticker} close price for tomorrow is: <b>{predicted_close_price:.2f} USD</b>",unsafe_allow_html=True)
        
    
    # Define the features and target
    features = ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Pct_Change']
    target = ['Close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.3, random_state=42)

    # Train a random forest regression model
    rf = RandomForestRegressor(n_estimators=200, random_state=1)
    rf.fit(X_train, y_train.values.ravel())

    # Define a function to predict tomorrow's close price
    def predict_next_close_price():
        # Get the last row of data
        last_row = data.tail(1)[features]

        # Predict tomorrow's close price
        predicted_close_price = rf.predict(last_row)[0]


    # Create a button to trigger the prediction
    if st.button("Random Forest ML prediction"):
        # Display the predicted close price
        st.write(f"Predicted {ticker} close price for tomorrow is: <b>{predicted_close_price:.2f} USD</b>",unsafe_allow_html=True)