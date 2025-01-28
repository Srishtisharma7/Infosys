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

API_KEY = "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"  # Groq API Key

SLACK_WEBHOOK = "https://hooks.slack.com/services/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"  # Slack webhook url

def truncate_text(text, max_length=512):
    return text[:max_length]

def load_competitor_data():
    """Load competitor data from a CSV file."""
    data = pd.read_csv("/Users/srishtisharma/python/infosys/competitor_data.csv")
    print(data.head())
    return data


def load_reviews_data():
    """Load reviews data from a csv file."""
    reviews = pd.read_csv("/Users/srishtisharma/python/infosys/reviews.csv")
    print(reviews.head)
    return reviews

def analyze_sentiment(reviews):
    """Analyze customer sentiment for reviews."""
    sentiment_pipeline = pipeline("sentiment-analysis")
    return sentiment_pipeline(reviews)

def train_predictive_model(data):
    """Train a predictive model for competitor pricing strategy."""
    data["Discount"] = data["Discount"].str.replace("%", "").astype(float)
    data["Price"] = data["Price"].astype(int)
    data["Predicted_Discount"] = data["Discount"] + (data["Price"] * 0.05).round(2)

    x = data[["Price", "Discount"]]
    y = data["Predicted_Discount"]
    print(x)

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, train_size=0.8
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train , y_train)
    return model



import numpy as np
import pandas as pd

def forecast_discounts_arima(data, future_days=5):
    """
    Forecast future discounts using ARIMA.

    Args:
        data: DataFrame containing historical discount data (with a datetime index).
        future_days: Number of days to forecast.

    Returns:
        DataFrame with historical and forecasted discounts.
    """

    data = data.sort_index()
    print(product_data.index)

    data["Discount"] = pd.to_numeric(data["Discount"], errors="coerce")
    data = data.dropna(subset=["Discount"])


    discount_series = data["Discount"]
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            raise ValueError("Index must be datetime or convertible to datetime."
                             ) from e



    model = ARIMA(discount_series, order=(5, 1, 0))
    model_fit = model.fit()


    forecast = model_fit.forecast(steps=future_days)
    future_dates = pd.date_range(
        start=discount_series.index[-1] + pd.Timedelta(days=1), periods=future_days
    )

    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Discount": forecast})
    forecast_df.set_index("Date", inplace=True)

    return forecast_df

def send_to_slack(data):
    """
    Sends a message to a Slack channel using the provided webhook URL.

    Args:
        data: The message to be sent to Slack.
    """
    payload = {"text": data}
    response = requests.post(
        SLACK_WEBHOOK,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )

def generate_strategy_recommendation(product_name, competitor_data, sentiment):
    """Generate strategic recommendations using an LLM."""

    date = datetime.now()
    prompt = f"""
    You are a highly skilled business strategist specializing in e-commerce. Based on the following details, suggest strategic recommendations:

    1. **Product Name**: {product_name}
    2. **Competitor Data** (including current prices, discounts, and predicted discounts):
    {competitor_data}
    3. **Sentiment Analysis**:
    {sentiment}
    4. **Today's Date**: {str(date)}

    ### Task:
    - Analyze the competitor data and identify key pricing trends.
    - Leverage sentiment analysis insights to highlight areas where customer satisfaction can be improved.
    - Use the discounted predictions to suggest how pricing strategies can be optimized over the next 5 days.
    - Recommend promotional campaigns or marketing strategies that align with customer sentiments.
    - Ensure the strategies are actionable , realistic , and geared toward increasing customer satisfaction , driving growth.

    Provide your recommendations in a structured format:
    1. **Pricing Strategy**
    2. **Promotional Campaign Ideas**
    3. **Customer Satisfaction Recommendations**

    """
    
    messages = [{"role": "user", "content": prompt}]

    data = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "llama3-8b-8192",
        "temperature": 0,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(data),
        headers=headers,
    )

    res_json = res.json()
    response = res_json["choices"][0]["message"]["content"]

    return response

############.........................#########################

st.set_page_config(page_title="E-Commerce Competitor Strategy Dashboard", layout="wide")


st.title("E-Commerce Competitor Strategy Dashboard")

st.sidebar.header("Select a Product")

products = [
    "Samsung Galaxy S24 Ultra Cell Phone, 512GB AI Smartphone" ,
    "Moto G Play | 2024 ",
    "Tracfone | Motorola Moto g Play 2024 | Locked | 64GB | 5000mAh Battery |",
    "OnePlus Nord N30 5G | Unlocked Dual-SIM Android Smart Phone"
]

selected_product = st.sidebar.selectbox("Choose a product to analyze:", products)

competitor_data = load_competitor_data()
reviews_data = load_reviews_data()

product_data = competitor_data[competitor_data["product_name"] == selected_product]
product_reviews = reviews_data[reviews_data["product_name"] == selected_product]

st.header(f"Competitor Analysis for {selected_product}")
st.subheader("Competitor Data")
st.table(product_data.tail(5))



if not product_reviews.empty:
    product_reviews["reviews"] = product_reviews["reviews"].apply(
        lambda x: truncate_text(x, 512)
    )
    reviews = product_reviews["reviews"].tolist()
    sentiments = analyze_sentiment(reviews)

    st.subheader("Customer Sentiment Analysis")
    sentiment_df = pd.DataFrame(sentiments)
    fig = px.bar(sentiment_df, x="label", title="Sentiment Analysis Results")
    st.plotly_chart(fig)
else:
    st.write("No reviews available for this product.")


# Preprocessing
product_data["Date"] = pd.to_datetime(product_data["Date"], errors="coerce")
product_data = product_data.dropna(subset=["Date"])
product_data.set_index("Date", inplace=True)
product_data = product_data.sort_index()

product_data["Discount"] = pd.to_numeric(product_data["Discount"], errors="coerce")
product_data = product_data.dropna(subset=["Discount"])

# Forecasting Model
product_data_with_predictions = forecast_discounts_arima(product_data)

st.subheader("Competitor Current and Predicted Discounts")
st.table(product_data_with_predictions.tail(10))

recommendations = generate_strategy_recommendation(
    selected_product,
    product_data_with_predictions,
    sentiments if not product_reviews.empty else "No reviews available",
)

st.subheader("Strategic Recommendations")
st.write(recommendations)

send_to_slack(recommendations)