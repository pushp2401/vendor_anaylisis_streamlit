import numpy as np
import joblib
import pandas as pd
# from tensorflow.keras.models import load_model
import tensorflow as tf
from openai import OpenAI 
from dotenv import load_dotenv 

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU'))) 
import keras
import os
import pickle
from gnews import GNews 
from textblob import TextBlob 
from pypdf import PdfReader 
load_dotenv() 

client = OpenAI(api_key= os.getenv('OPENAI_API_KEY')) 


cwd = os.getcwd() 

################## bankruptcy module #####################

def predict_bankruptcy(dictt) :
    # with open("/Users/pushpanjali/price_model/bankruptcy/app/models/regression_model.pkl", "rb") as f:
    with open(cwd + "/models/regression_model.pkl", "rb") as f:
        loaded_model_bankruptcy = pickle.load(f)
    # scaler_bankruptcy = joblib.load("/Users/pushpanjali/price_model/bankruptcy/app/models/bankruptcy_model_input_scaler.pkl") 
    scaler_bankruptcy = joblib.load(cwd + "/models/bankruptcy_model_input_scaler.pkl") 

    input_array = np.array(list(dictt.values())).reshape(1 , -1)
    scaled_test_array = scaler_bankruptcy.transform(input_array) 
    prediction = loaded_model_bankruptcy.predict(scaled_test_array) 
    return prediction


################## aluminium price predictor module #################


def predict_price(date1, vol1, price_change1, open1, high1, low1, df, time_steps=30):
    """
    Predicts the price for a given date using an LSTM model.

    Parameters:
    - date1 (str): Date for prediction (YYYY-MM-DD)
    - vol1 (float): Volume on that date
    - price_change1 (float): Price change on that date
    - open1 (float): Open price on that date
    - high1 (float): High price on that date
    - low1 (float): Low price on that date
    - df (DataFrame): Historical data (must include 'datetime' index)
    - time_steps (int): Number of past time steps to consider for LSTM

    Returns:
    - Predicted price (float)
    """
    # import pdb; pdb.set_trace()
    # loaded_model = tf.keras.models.load_model('/Users/pushpanjali/price_model/bankruptcy/app/models/al_price_predictor.keras' , compile = False) 
    loaded_model = tf.keras.models.load_model(cwd + '/models/al_price_predictor.keras' , compile = False) 
    # loaded_model.compile(optimizer="adam", loss="mae")
    # scaler = joblib.load("/Users/pushpanjali/price_model/bankruptcy/app/models/al_model_input_scaler.pkl")  # Ensure it's the same scaler used during training
    scaler = joblib.load(cwd + "/models/al_model_input_scaler.pkl")  # Ensure it's the same scaler used during training

    # 🔹 Ensure date is in datetime format
    df.index = pd.to_datetime(df.index)

    # 🔹 Check if the given date exists in the dataset
    if date1 not in df.index:
        raise ValueError(f"❌ Date {date1} not found in dataset!")

    # 🔹 Select last `time_steps - 1` values before `date1`
    X_past = df.loc[:date1, ['Vol.', 'price_change', 'Open', 'High', 'Low', 'Price']].tail(time_steps - 1).values
    # print(X_past)
    # 🔹 Append new row (date1 values) to maintain `time_steps` shape
    X_input = np.vstack([X_past, [vol1, price_change1, open1, high1, low1, 0]])  # `0` as placeholder for Price

    # 🔹 Scale input using the same scaler
    X_input_scaled = scaler.transform(X_input)
    # print(X_input_scaled)
    # 🔹 Remove the last column (Price) before passing to LSTM
    X_input_scaled = X_input_scaled[:, :-1]  # Remove target column (Price)

    # 🔹 Reshape input for LSTM (samples, time_steps, features)
    X_input_scaled = np.expand_dims(X_input_scaled, axis=0)  # Shape: (1, time_steps, features)
    # print(X_input_scaled)
    # 🔹 Make prediction (scaled output)
    if np.isnan(X_input_scaled).any():
        X_input_scaled = np.nan_to_num(X_input_scaled, nan=0)  # Replace NaN with 0

        print("replaced nan with 0 \n")
        # raise ValueError("❌ Error: Scaled input contains NaN values! Check preprocessing steps.")

    predicted_price_scaled = loaded_model.predict(X_input_scaled)
    # print(predicted_price_scaled)
    # 🔹 Create placeholder array to inverse transform
    placeholder = np.zeros((1, X_input.shape[1]))  # Shape: (1, num_features)
    placeholder[0, -1] = predicted_price_scaled  # Set predicted Price in last column

    # 🔹 Reverse transform the prediction using the same scaler
    predicted_price = scaler.inverse_transform(placeholder)[0, -1]

    return round(predicted_price, 2)

def news_fetcher(company , days) :
    google_news = GNews( period= days)
    news = google_news.get_news(company)
    # print(len(tata_news))
    news_df = pd.DataFrame(news)
    news_df["Text"] = news_df['title'] + news_df['description'] 
    return news_df

def analyze_sentiment(df):
    """Compute sentiment scores for a given dataframe."""
    df["Sentiment"] = df["Text"].apply(lambda text: TextBlob(text).sentiment.polarity)
    avg_sentiment = df["Sentiment"].mean()
    if avg_sentiment > 0.2 :
        rating = "Positive"
    if avg_sentiment < 0.2 and avg_sentiment > -0.2 :
        rating = "Neutral"
    if avg_sentiment < -0.2 :
        rating = "Negative"
    
    # rating = "Positive" if avg_sentiment > 0.2 else "Negative" if avg_sentiment < -0.2 else "Neutral" 
    return avg_sentiment, rating

def highlight_big_news(df):
    """Highlight articles with extreme sentiment."""
    big_news = df[abs(df["Sentiment"]) > 0.7]
    big_news_para = ""
    for i in big_news.index : 
        para = big_news.loc[i , 'published date'] + " : " + big_news.loc[ i  , "Text"] + "\n"
        big_news_para += para
    return big_news_para


# 🔹 Load dataset (modify file path as needed)
# df = pd.read_csv("your_data.csv", parse_dates=["datetime"], index_col="datetime")

# 🔹 Predict price for a given date
# predicted_price = predict_price(
#     date1="2024-08-29",
#     vol1=10,
#     price_change1=-1.62,
#     open1=230.15,
#     high1=230.15,
#     low1=227.5,
#     df= training_data
# )

# print(f"\n📅 Predicted Price on 2024-08-29: {predicted_price}\n") 

def tender_details_gpt(text) : 
    resp = client.chat.completions.create(
        model = "gpt-4o-mini",
        response_format={"type" : "json_object"} , 
        messages=[
            {
                "role" : "system" , "content" : """
You are a helpful construction assistant. Your task is to extract following details from a given tender document and output in JSON format. 
Project name
Project cost
Project timeline
Project type
Project starting date
Project ending date
Steel quantity to be used
Cement quantity to be used
Project size (cost based)
Project size(duration based)
"""
            },
            {
                "role" : "user" , "content" : f"Here is the tender document : {text}"
            }
        ] 
    )
    return resp.choices[0].message.content.strip() 


def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file using PyMuPDF."""
    # doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    # reader = PdfReader("/Users/pushpanjali/price_model/bankruptcy/app/input/sample_financials.pdf")
    reader = PdfReader(pdf_file) 

    text = "".join(page.extract_text("") for page in reader.pages)
    return text

def query_gpt(prompt):
    """Send extracted text to OpenAI's GPT model and get a response."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type" : "json_object"} , 
        messages=[{"role": "system", "content": """You are an AI that extracts following financial metrices in below order from given document and returns them strictly in JSON format.                  
        ROA(C) before interest and depreciation before interest
        ROA(A) before interest and % after tax
        ROA(B) before interest and depreciation after tax
        Operating Gross Margin
        Realized Sales Gross Margin
        Operating Profit Rate
        Pre-tax net Interest Rate
        After-tax net Interest Rate
        Non-industry income and expenditure/revenue
        Continuous interest rate (after tax)
        Operating Expense Rate
        Research and development expense rate
        Cash flow rate
        Interest-bearing debt interest rate
        Tax rate (A)
        Net Value Per Share (B)
        Net Value Per Share (A)
        Net Value Per Share (C)
        Persistent EPS in the Last Four Seasons
        Cash Flow Per Share
        Revenue Per Share
        Operating Profit Per Share
        Per Share Net profit before tax
        Realized Sales Gross Profit Growth Rate
        Operating Profit Growth Rate
        After-tax Net Profit Growth Rate
        Regular Net Profit Growth Rate
        Continuous Net Profit Growth Rate
        Total Asset Growth Rate
        Net Value Growth Rate
        Total Asset Return Growth Rate Ratio
        Cash Reinvestment %
        Current Ratio
        Quick Ratio
        Interest Expense Ratio
        Total debt/Total net worth
        Debt ratio %
        Net worth/Assets
        Long-term fund suitability ratio (A)
        Borrowing dependency
        Contingent liabilities/Net worth
        Operating profit/Paid-in capital
        Net profit before tax/Paid-in capital
        Inventory and accounts receivable/Net value
        Total Asset Turnover
        Accounts Receivable Turnover
        Average Collection Days
        Inventory Turnover Rate (times)
        Fixed Assets Turnover Frequency
        Net Worth Turnover Rate (times)
        Revenue per person
        Operating profit per person
        Allocation rate per person
        Working Capital to Total Assets
        Quick Assets/Total Assets
        Current Assets/Total Assets
        Cash/Total Assets
        Quick Assets/Current Liability
        Cash/Current Liability
        Current Liability to Assets
        Operating Funds to Liability
        Inventory/Working Capital
        Inventory/Current Liability
        Current Liabilities/Liability
        Working Capital/Equity
        Current Liabilities/Equity
        Long-term Liability to Current Assets
        Retained Earnings to Total Assets
        Total income/Total expense
        Total expense/Assets
        Current Asset Turnover Rate
        Quick Asset Turnover Rate
        Working capitcal Turnover Rate
        Cash Turnover Rate
        Cash Flow to Sales
        Fixed Assets to Assets
        Current Liability to Liability
        Current Liability to Equity
        Equity to Long-term Liability
        Cash Flow to Total Assets
        Cash Flow to Liability
        CFO to Assets
        Cash Flow to Equity
        Current Liability to Current Assets
        Liability-Assets Flag
        Net Income to Total Assets
        Total assets to GNP price
        No-credit Interval
        Gross Profit to Sales
        Net Income to Stockholder's Equity
        Liability to Equity
        Degree of Financial Leverage (DFL)
        Interest Coverage Ratio (Interest expense to EBIT)
        Net Income Flag
        Equity to Liability"""},
                  {"role": "user", "content": prompt}]
    )
    # print(response)
    # f = response.choices[0].message.parsed
    # print(f)
    # print(type(f))
    return response.choices[0].message.content.strip() 


