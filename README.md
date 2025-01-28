# E-Commerce Competitor Strategy Dashboard

This project analyzes e-commerce competitors' data, performs sentiment analysis on customer reviews, and provides strategic recommendations for pricing and discount strategies using machine learning and forecasting techniques.

### The dashboard allows users to:

Analyze competitor pricing and discount strategies.

Visualize customer sentiment analysis from product reviews.

Forecast future discount trends.

Generate strategic recommendations based on competitor and sentiment data.

Integrate with Slack for sending real-time recommendations.


## Features

Competitor Data Analysis: Displays competitor prices, discounts, and predicts future discounts.

Sentiment Analysis: Analyzes customer sentiment from product reviews using a Hugging Face model.

Discount Forecasting: Uses ARIMA to forecast discounts for the next few days.

Strategic Recommendations: Generates business recommendations based on competitor data and sentiment analysis using a large language model (LLM).

Slack Integration: Sends strategic recommendations to a Slack channel for team collaboration.


## Prerequisites

Before running the project, make sure you have the following:

Python 3.7 or higher

Install the required libraries. You can install them using pip:
pip install pandas plotly requests openai streamlit scikit-learn statsmodels transformers


## Clone the project to your local machine:

git clone https://github.com/yourusername/ecommerce-competitor-strategy-dashboard.git
cd ecommerce-competitor-strategy-dashboard

### 2. Prepare the Data Files
Make sure the data files (competitor_data.csv and reviews.csv) are placed in the same directory as the Python scripts. These files should contain:

competitor_data.csv: Data with product names, prices, discounts, and other competitor pricing info.
reviews.csv: Data containing product reviews.


### 3. Run the Streamlit Dashboard
To start the Streamlit dashboard, run the following command in the terminal:

streamlit run streamlit_dashboard.py
This will start the dashboard locally, and you can access it by navigating to http://localhost:8501 in your browser.

### 4. Slack Webhook Setup
To integrate Slack, you'll need a webhook URL. Follow these steps to create one:

Go to Slack Incoming Webhooks.
Create a new webhook and copy the URL.
Paste the webhook URL in the SLACK_WEBHOOK variable inside the Python scripts.

### 5. API Key Setup
For generating strategic recommendations, you need an API key for the Groq API:

Sign up at Groq API.
Get your API key and paste it in the API_KEY variable in the Python scripts.


## How It Works

### Competitor Data Analysis:
Competitor pricing and discount data is loaded, cleaned, and processed.
A predictive model is trained using the competitor's prices and discounts to predict future pricing trends.
### Sentiment Analysis:
Customer reviews are loaded and sentiment is analyzed using Hugging Face's sentiment-analysis model.
The results are visualized with a Plotly bar chart, showing the sentiment distribution (positive, neutral, negative).
### Discount Forecasting:
The ARIMA model is used to forecast future discount trends for the selected product.
A table of predicted discounts is displayed.
### Strategic Recommendations:
The strategic recommendations are generated based on competitor data, sentiment analysis, and AI predictions.
Recommendations are sent to a Slack channel via webhook for real-time collaboration.


#### Feel free to fork the repository and create a pull request if you'd like to contribute. For any bug fixes or improvements, create an issue and we can work on it together.

### License

This project is licensed under the MIT License - see the LICENSE file for details.



!Example(https://github.com/user-attachments/assets/09d34708-6d10-4345-9b85-fa144c819b16)
!DASHBOARD](https://github.com/user-attachments/assets/e628caa5-1804-4e1f-ba7c-0504f8134337)
