import numpy as np
import joblib
import pandas as pd
# from tensorflow.keras.models import load_model
import tensorflow as tf
from openai import OpenAI 
from dotenv import load_dotenv 
import streamlit as st
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU'))) 
import keras
import os
import pickle
from gnews import GNews 
from textblob import TextBlob 
from pypdf import PdfReader 
load_dotenv() 
from datetime import datetime

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

def predict_bankruptcy_selected_feature(dictt) :
    # with open("/Users/pushpanjali/price_model/bankruptcy/app/models/regression_model.pkl", "rb") as f:
    # with open(cwd + "/models/regression_model_selected_features.pkl", "rb") as f:
    # with open(cwd + "/models/regression_model_selected_features_50_50.pkl", "rb") as f:
    # with open(cwd + "/models/regression_model_selected_features_50_50.pkl")
        # loaded_model_bankruptcy = pickle.load(f)
    loaded_model_bankruptcy = joblib.load(cwd + "/models/xgb_model.pkl")
    # scaler_bankruptcy = joblib.load("/Users/pushpanjali/price_model/bankruptcy/app/models/bankruptcy_model_input_scaler.pkl") 
    # scaler_bankruptcy = joblib.load(cwd + "/models/bankruptcy_model_input_scaler_selected_features.pkl") 
    # scaler_bankruptcy = joblib.load(cwd + "/models/bankruptcy_model_input_scaler_selected_features_50_50.pkl") 
    scaler_bankruptcy = joblib.load(cwd + "/models/robust_scaler.pkl") 

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
    date_object = datetime.strptime(date1 , "%Y-%m-%d")
    try : 
        if date_object not in df.index:
            print(f"❌ Date {date1} not found in dataset!")
            date1 = "2024-08-29" 
        if date_object in df.index :
            date1 = date_object
    except :
        print("switching to hardcoded date")
        date1 = "2024-08-29" 


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
    if len(news) > 0 :
    # print(len(tata_news))
        news_df = pd.DataFrame(news)
        news_df["Text"] = news_df['title'] + news_df['description'] 
    else :
        news_df = pd.DataFrame()
        st.write("No news found related to this company")
    return news_df
    

def analyze_sentiment(df):
    """Compute sentiment scores for a given dataframe."""
    if len(df) > 0 :
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
    else :
        return "Not available" , "Not available"

def highlight_big_news(df):
    """Highlight articles with extreme sentiment."""
    if len(df) > 0 :
        big_news = df[abs(df["Sentiment"]) > 0.7]
        big_news_para = ""
        for i in big_news.index : 
            para = big_news.loc[i , 'published date'] + " : " + big_news.loc[ i  , "Text"] + "\n"
            big_news_para += para
        return big_news_para
    else :
        return "No big news available for this company"


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
        temperature= 0,
        response_format={"type" : "json_object"} , 
        messages=[
            {
                "role" : "system" , "content" : """
You are a helpful construction assistant. Your task is to extract following details according to mentioned datatype from a given tender document and output in JSON format.
If any details is not available keep the value as "NA" for them .
"Project name" : string,
"Project cost($)" : int,
"Project timeline" : string,
"Project type" : string,
"Project starting date" : "yyyy-mm-dd",
"Project ending date" : "yyyy-mm-dd",
"Steel quantity to be used in Metric Ton" : int,
"Cement/Concrete quantity to be used in MTetric Ton" : int,
"Project size (cost based)" : str,
"Project size(duration based) : str"
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
        # model = 'o1',
        temperature=0,
        response_format={"type" : "json_object"} , 
        messages=[{"role": "system", "content": """you are a smart finanical assitant. your task is given a finanical document extract the metrices from the document that are listed below and output in JSON format.
        For the metrices that are not listed calculate them from available financial metrices. 
        If any metrics is not available keep their value as 0.                 
        " ROA(C) before interest and depreciation before interest" : float,
        " ROA(A) before interest and % after tax" : float,
        " ROA(B) before interest and depreciation after tax" : float,
        " Operating Gross Margin" : float,
        " Realized Sales Gross Margin" : float,
        " Operating Profit Rate" : float,
        " Pre-tax net Interest Rate" : float,
        " After-tax net Interest Rate" : float,
        " Non-industry income and expenditure/revenue" : float,
        " Continuous interest rate (after tax)" : float,
        " Operating Expense Rate" : float,
        " Research and development expense rate" : float,
        " Cash flow rate" : float,
        " Interest-bearing debt interest rate" : float,
        " Tax rate (A)" : float,
        " Net Value Per Share (B)" : float,
        " Net Value Per Share (A)" : float,
        " Net Value Per Share (C)" : float,
        " Persistent EPS in the Last Four Seasons" : float,
        " Cash Flow Per Share" : float,
        " Revenue Per Share" : float,
        " Operating Profit Per Share" : float,
        " Per Share Net profit before tax" : float,
        " Realized Sales Gross Profit Growth Rate" : float,
        " Operating Profit Growth Rate" : float,
        " After-tax Net Profit Growth Rate" : float,
        " Regular Net Profit Growth Rate" : float,
        " Continuous Net Profit Growth Rate" : float,
        " Total Asset Growth Rate" : float,
        " Net Value Growth Rate" : float,
        " Total Asset Return Growth Rate Ratio" : float,
        " Cash Reinvestment %" : float,
        " Current Ratio" : float,
        " Quick Ratio" : float,
        " Interest Expense Ratio" : float,
        " Total debt/Total net worth" : float,
        " Debt ratio %" : float,
        " Net worth/Assets" : float,
        " Long-term fund suitability ratio (A)" : float,
        " Borrowing dependency" : float,
        " Contingent liabilities/Net worth" : float,
        " Operating profit/Paid-in capital" : float,
        " Net profit before tax/Paid-in capital" : float,
        " Inventory and accounts receivable/Net value" : float,
        " Total Asset Turnover" : float,
        " Accounts Receivable Turnover" : float,
        " Average Collection Days" : float,
        " Inventory Turnover Rate (times)" : float,
        " Fixed Assets Turnover Frequency" : float,
        " Net Worth Turnover Rate (times)" : float,
        " Revenue per person" : float,
        " Operating profit per person" : float,
        " Allocation rate per person" : float,
        " Working Capital to Total Assets" : float,
        " Quick Assets/Total Assets" : float,
        " Current Assets/Total Assets" : float,
        " Cash/Total Assets" : float,
        " Quick Assets/Current Liability" : float,
        " Cash/Current Liability" : float,
        " Current Liability to Assets" : float,
        " Operating Funds to Liability" : float,
        " Inventory/Working Capital" : float,
        " Inventory/Current Liability" : float,
        " Current Liabilities/Liability" : float,
        " Working Capital/Equity" : float,
        " Current Liabilities/Equity" : float,
        " Long-term Liability to Current Assets" : float,
        " Retained Earnings to Total Assets" : float,
        " Total income/Total expense" : float,
        " Total expense/Assets" : float,
        " Current Asset Turnover Rate" : float,
        " Quick Asset Turnover Rate" : float,
        " Working capitcal Turnover Rate" : float,
        " Cash Turnover Rate" : float,
        " Cash Flow to Sales" : float,
        " Fixed Assets to Assets" : float,
        " Current Liability to Liability" : float,
        " Current Liability to Equity" : float,
        " Equity to Long-term Liability" : float,
        " Cash Flow to Total Assets" : float,
        " Cash Flow to Liability" : float,
        " CFO to Assets" : float,
        " Cash Flow to Equity" : float,
        " Current Liability to Current Assets" : float,
        " Liability-Assets Flag" : float,
        " Net Income to Total Assets" : float,
        " Total assets to GNP price" : float,
        " No-credit Interval" : float,
        " Gross Profit to Sales" : float,
        " Net Income to Stockholder's Equity" : float,
        " Liability to Equity" : float,
        " Degree of Financial Leverage (DFL)" : float,
        " Interest Coverage Ratio (Interest expense to EBIT)" : float,
        " Net Income Flag" : float,
        " Equity to Liability" : float,"""},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip() 

def query_gpt_selected_feature(prompt):
    """Send extracted text to OpenAI's GPT model and get a response."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        # model = 'o1',
        temperature=0,
        response_format={"type" : "json_object"} , 
        messages=[{"role": "system", "content": """you are a smart finanical assitant. your task is given a finanical document extract the metrices from the document that are listed below and output in JSON format.
        For the metrices that are not listed calculate them from available financial metrices. 
        If any metrics is not available keep their value as 0.                 
        "Operating Gross Margin" : float,
        "Realized Sales Gross Margin" : float,
        "Operating Profit Rate" : float,
        "Operating Expense Rate" : float,
        "Current Ratio" : float,
        "Debt ratio %" : float,
        "Gross Profit to Sales" : float"""},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip() 

def query_gpt_xgb(prompt):
    """Send extracted text to OpenAI's GPT model and get a response."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        # model = 'o1',
        temperature=0,
        response_format={"type" : "json_object"} , 
        messages=[{"role": "system", "content": """you are a smart finanical assitant. your task is given a finanical document extract the metrices from the document that are listed below and output in JSON format.
        For the metrices that are not listed calculate them from available financial metrices otherwise, return "NA"                
        "current assests" : float ,
        "cost of good sold" : float , 
        "Depreciation and amortization" : float ,
        "EBITDA" : float ,
        "Inventory" : float,
        "Net Income" : float ,
        "Total Receivables" : float , 
        "Net sales" : float , 
        "Total assests" : float , 
        "Total Long-term debt " : float ,
        "EBIT" : float , 
        "Gross Profit" : float , 
        "Total Current Liabilities" : float , 
        "Retained Earnings" : float , 
        "Total Revenue" : float , 
        "Total Liabilities " : float , 
        "Total Operating Expenses" : float 
                   """},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip() 


# def query_gpt(prompt):
#     """Send extracted text to OpenAI's GPT model and get a response."""
#     response = client.chat.completions.create(
#         model="gpt-4o",
#         temperature=0,
#         response_format={"type" : "json_object"} , 
#         messages=[{"role": "system", "content": """you are a smart finanical assitant. your task is given a finanical document extract the metrices from the document that are listed below and return in JSON format.
#         For the metrices that are not listed calculate them from available financial metrices.                  
#         " ROA(C) before interest and depreciation before interest" : float,
#         " ROA(A) before interest and % after tax" : float,
#         " ROA(B) before interest and depreciation after tax" : float,
#         " Operating Gross Margin" : float,
#         " Realized Sales Gross Margin" : float,
#         " Operating Profit Rate" : float,
#         " Pre-tax net Interest Rate" : float,
#         " After-tax net Interest Rate" : float,
#         " Non-industry income and expenditure/revenue" : float,
#         " Continuous interest rate (after tax)" : float,
#         " Operating Expense Rate" : float,
#         " Research and development expense rate" : float,
#         " Cash flow rate" : float,
#         " Interest-bearing debt interest rate" : float,
#         " Tax rate (A)" : float,
#         " Net Value Per Share (B)" : float,
#         " Net Value Per Share (A)" : float,
#         " Net Value Per Share (C)" : float,
#         " Persistent EPS in the Last Four Seasons" : float,
#         " Cash Flow Per Share" : float,
#         " Revenue Per Share" : float,
#         " Operating Profit Per Share" : float,
#         " Per Share Net profit before tax" : float,
#         " Realized Sales Gross Profit Growth Rate" : float,
#         " Operating Profit Growth Rate" : float,
#         " After-tax Net Profit Growth Rate" : float,
#         " Regular Net Profit Growth Rate" : float,
#         " Continuous Net Profit Growth Rate" : float,
#         " Total Asset Growth Rate" : float,
#         " Net Value Growth Rate" : float,
#         " Total Asset Return Growth Rate Ratio" : float,
#         " Cash Reinvestment %" : float,
#         " Current Ratio" : float,
#         " Quick Ratio" : float,
#         " Interest Expense Ratio" : float,
#         " Total debt/Total net worth" : float,
#         " Debt ratio %" : float,
#         " Net worth/Assets" : float,
#         " Long-term fund suitability ratio (A)" : float,
#         " Borrowing dependency" : float,
#         " Contingent liabilities/Net worth" : float,
#         " Operating profit/Paid-in capital" : float,
#         " Net profit before tax/Paid-in capital" : float,
#         " Inventory and accounts receivable/Net value" : float,
#         " Total Asset Turnover" : float,
#         " Accounts Receivable Turnover" : float,
#         " Average Collection Days" : float,
#         " Inventory Turnover Rate (times)" : float,
#         " Fixed Assets Turnover Frequency" : float,
#         " Net Worth Turnover Rate (times)" : float,
#         " Revenue per person" : float,
#         " Operating profit per person" : float,
#         " Allocation rate per person" : float,
#         " Working Capital to Total Assets" : float,
#         " Quick Assets/Total Assets" : float,
#         " Current Assets/Total Assets" : float,
#         " Cash/Total Assets" : float,
#         " Quick Assets/Current Liability" : float,
#         " Cash/Current Liability" : float,
#         " Current Liability to Assets" : float,
#         " Operating Funds to Liability" : float,
#         " Inventory/Working Capital" : float,
#         " Inventory/Current Liability" : float,
#         " Current Liabilities/Liability" : float,
#         " Working Capital/Equity" : float,
#         " Current Liabilities/Equity" : float,
#         " Long-term Liability to Current Assets" : float,
#         " Retained Earnings to Total Assets" : float,
#         " Total income/Total expense" : float,
#         " Total expense/Assets" : float,
#         " Current Asset Turnover Rate" : float,
#         " Quick Asset Turnover Rate" : float,
#         " Working capitcal Turnover Rate" : float,
#         " Cash Turnover Rate" : float,
#         " Cash Flow to Sales" : float,
#         " Fixed Assets to Assets" : float,
#         " Current Liability to Liability" : float,
#         " Current Liability to Equity" : float,
#         " Equity to Long-term Liability" : float,
#         " Cash Flow to Total Assets" : float,
#         " Cash Flow to Liability" : float,
#         " CFO to Assets" : float,
#         " Cash Flow to Equity" : float,
#         " Current Liability to Current Assets" : float,
#         " Liability-Assets Flag" : float,
#         " Net Income to Total Assets" : float,
#         " Total assets to GNP price" : float,
#         " No-credit Interval" : float,
#         " Gross Profit to Sales" : float,
#         " Net Income to Stockholder's Equity" : float,
#         " Liability to Equity" : float,
#         " Degree of Financial Leverage (DFL)" : float,
#         " Interest Coverage Ratio (Interest expense to EBIT)" : float,
#         " Net Income Flag" : float,
#         " Equity to Liability" : float,,}"""},
#                   {"role": "user", "content": prompt}]
#     )
#     # print(response)
#     # f = response.choices[0].message.parsed
#     # print(f)
#     # print(type(f))
#     return response.choices[0].message.content.strip() 


