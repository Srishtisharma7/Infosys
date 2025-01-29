import pandas as pd
import plotly.express as px
import requests
import json
from datetime import datetime
import streamlit as st
from openai import AzureOpenAI
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from transformers import pipeline

# Set up the Streamlit dashboard configuration
st.set_page_config(page_title="E-Commerce Competitor Strategy Dashboard", layout="wide")

# Define the API keys and Webhook URL for Groq API and Slack integration
API_KEY = "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"

SLACK_WEBHOOK = "https://hooks.slack.com/services/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"  # Slack webhook url

def truncate_text(text, max_length=512):
    return text[:max_length]

# load competitor pricing data from a CSV file
def load_competitor_data():
    """Load competitor data from a CSV file."""
    data = pd.read_csv("competitor_data.csv")
    print(data.head())
    return data

# load reviews data from a CSV file
def load_reviews_data():
    """Load reviews data from a csv file."""
    reviews = pd.read_csv("reviews.csv")

    return reviews

# Function to analyze customer sentiment using sentiment-analysis pipeline
def analyze_sentiment(reviews):
    """Analyze customer sentiment for reviews."""
    sentiment_pipeline = pipeline("sentiment-analysis")
    return sentiment_pipeline(reviews)

def train_predictive_model(data):
    """Train a predictive model for competitor pricing strategy."""
    data["Discount"] = data["Discount"].str.replace("%", "").astype(float)
    data["Price"] = data["Price"].astype(int)
    data["Predicted_Discount"] = data["Discount"] + (data["Price"] * 0.05).round(2)
    
 # Define features (X) and target (y) for training the model
    x = data[["Price", "Discount"]]
    y = data["Predicted_Discount"]
    print(x)

    # Split data into training and testing sets for model validation
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, train_size=0.8
    )


import numpy as np
import pandas as pd

# Function to forecast future discount trends using ARIMA model
def forecast_discounts_arima(data, future_days=5):
 """Forecast discounts for the next 'future_days' using ARIMA."""
    data = data.sort_index() # Ensure data is sorted by date
    print(product_data.index)
    data["Discount"] = pd.to_numeric(data["Discount"], errors="coerce") # Handle invalid values in Discount
    data = data.dropna(subset=["Discount"])
    discount_series = data["Discount"]
   
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index) # Convert index to datetime
        except Exception as e:
            raise ValueError("Index must be datetime or convertible to datetime.") from e

    model = ARIMA(discount_series, order=(5, 1, 0)) # Define the ARIMA model parameter
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=future_days) # Forecast the next 'future_days' discount
    future_dates = pd.date_range(
        start=discount_series.index[-1] + pd.Timedelta(days=1), periods=future_days
    )
 # Create a DataFrame with forecasted data
    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Discount": forecast})
    forecast_df.set_index("Date", inplace=True)

    return forecast_df

# Function to send the generated strategy recommendation to Slack
def send_to_slack(data):
    """
    Sends a message to a Slack channel using the provided webhook URL.

    Args:
        data: The message to be sent to Slack.
    """
    payload = {"text": data} # Prepare the payload with message text
    response = requests.post(
        SLACK_WEBHOOK,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
# Post the message to Slack

# Function to generate strategic recommendations using a Large Language Model (LLM)
def generate_strategy_recommendation(product_name, competitor_data, sentiment):
    """Generate strategic recommendations using an LLM."""
    
    # Truncate competitor data to limit token usage
    competitor_data_summary = competitor_data.head(5)  # Only take first 5 rows for simplicity
    competitor_data_summary = competitor_data_summary.to_string(index=False)  # Convert to string for API
    
    # Truncate sentiment analysis to only relevant parts
    sentiment_summary = str(sentiment)[:512]  # Limit sentiment string to 512 characters
    
    # Prepare the prompt by reducing token usage
    date = datetime.now()
    prompt = f"""
    You are a skilled business strategist specializing in e-commerce. Based on the following details, suggest strategic recommendations:

    1. **Product Name**: {product_name}
    2. **Competitor Data** (prices, discounts, predicted discounts):
    {competitor_data_summary}
    3. **Sentiment Analysis**:
    {sentiment_summary}
    4. **Today's Date**: {str(date)}

    ### Task:
    - Analyze the competitor data and identify key pricing trends.
    - Leverage sentiment insights to suggest areas for improvement.
    """
    
    # Request data for Groq API
    data = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "llama3-8b-8192",
        "temperature": 0,
        "max_tokens": 512  # Limit tokens returned by the model
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    # Make the API request
    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(data),
        headers=headers,
    )

    res_json = res.json()
    print("APIsw response:", res_json)  # Check the full response for debugging

    # Check if response contains the 'choices' key and return the content
    if "choices" in res_json:
        response = res_json["choices"][0]["message"]["content"]
    else:
        response = "Error: 'choices' key is missing in the response. Please check the API request."

    return response


# Streamlit application code to build the dashboard interface
st.title("E-Commerce Competitor Strategy Dashboard")

st.sidebar.header("Select a Product") # sidebar 

products = [
    "Motorola razr | 2023 | Unlocked | Made for US 8/128 | 32MP Camera | Sage Green, 73.95 x 170.82 x 7.35mm",
    "Moto G Power 5G | 2024 | Unlocked | Made for US 8+128GB | 50MP Camera | Pale Lilac",
    "Tracfone | Motorola Moto g Play 2024 | Locked | 64GB | 5000mAh Battery | 50MP Quad Pixel Camera | 6.5-in. HD+ 90Hz Display | Sapphire Blue",
    "Samsung Galaxy S24 Ultra Cell Phone, 512GB AI Smartphone, Unlocked Android, 200MP, 100x Zoom Cameras, Fast Processor, Long Battery Life, Edge-to-Edge Display, S Pen, US Version, 2024, Titanium Black"
]

selected_product = st.sidebar.selectbox("Choose a product to analyze:", products)
# Load competitor data and reviews data
competitor_data = load_competitor_data()
reviews_data = load_reviews_data()

# Filter the selected product's data from the datasets
product_data = competitor_data[competitor_data["product_name"] == selected_product]
product_reviews = reviews_data[reviews_data["product_name"] == selected_product]


st.header(f"Competitor Analysis for {selected_product}")   # Display product's competitor data on the main page
st.table(product_data.tail(5))


# Analyze sentiment
if not product_reviews.empty:
    product_reviews["reviews"] = product_reviews["reviews"].apply(
        lambda x: truncate_text(x, 512)
    )
    reviews = product_reviews["reviews"].tolist()   # convert to list
    sentiments = analyze_sentiment(reviews)  # Get sentiment analysis for reviews

    st.subheader("Customer Sentiment Analysis")
    sentiment_df = pd.DataFrame(sentiments)
    fig = px.bar(sentiment_df, x="label", title="Sentiment Analysis Results")
    st.plotly_chart(fig)   # Display sentiment analysis results using Plotly bar chart
    
else:
    st.write("No reviews available for this product.")


# Preprocessing for forecasting
product_data["Date"] = pd.to_datetime(product_data["Date"], errors="coerce")
product_data = product_data.dropna(subset=["Date"])
product_data.set_index("Date", inplace=True)
product_data = product_data.sort_index()

product_data["Discount"] = pd.to_numeric(product_data["Discount"], errors="coerce")
product_data = product_data.dropna(subset=["Discount"])

# Forecasting Model to predict future discounts
product_data_with_predictions = forecast_discounts_arima(product_data)

# Display current and predicted discounts in a table
st.subheader("Competitor Current and Predicted Discounts")
st.table(product_data_with_predictions.tail(10))


# Generate strategic recommendations using the LLM
recommendations = generate_strategy_recommendation(
    selected_product,
    product_data_with_predictions,
    sentiments if not product_reviews.empty else "No reviews available",
)
# Display recommendations
st.subheader("Strategic Recommendations")
st.write(recommendations)

send_to_slack(recommendations)# Send the recommendations to Slack for further action
