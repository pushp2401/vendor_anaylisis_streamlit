import streamlit as st
import os
from dotenv import load_dotenv 
from pypdf import PdfReader 
from openai import OpenAI 
import json
import pandas as pd

# from st_app import extract_text_from_pdf , query_gpt , tender_details_gpt , client
from custom_functions import predict_bankruptcy , predict_price
from custom_functions import news_fetcher , analyze_sentiment , highlight_big_news , tender_details_gpt , extract_text_from_pdf , query_gpt , tender_details_gpt 

cwd = os.getcwd() 


# Load environment variables
load_dotenv() 
client = OpenAI(api_key= os.getenv('OPENAI_API_KEY')) 
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
        temperature= 0 , 
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
    return response.choices[0].message.content.strip() 


st.title("Construction Vendor Bankruptcy Analysis")

uploaded_files = st.file_uploader("Upload multiple PDF files", type=["pdf" , "csv"], accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:

        # st.subheader(f"Processing: {uploaded_file.name}") 
        import pdb; pdb.set_trace()
        if uploaded_file.name == "2_financials.pdf" :
            st.subheader(f"Processing: {uploaded_file.name}")

            with st.spinner("Accessing bankruptcy" , show_time = True) :
                # st.subheader(f"Processing: {uploaded_file.name}")

                extracted_text = extract_text_from_pdf(uploaded_file)
                metrics_dict = json.loads(query_gpt(extracted_text)) 
                bankruptcy_status = predict_bankruptcy(metrics_dict)
                if bankruptcy_status[0] == 0 : 
                    st.text_area("Bankruptcy status prediction" ,value = f"Predicted bankruptcy status of company: Not bankrupt" , key = "bsp0")
                if bankruptcy_status[0] == 1 : 
                    st.text_area("Bankruptcy status prediction" ,value = f"Predicted bankruptcy status of company: Bankrupt", key = "bsp1")
        if uploaded_file.name == "1_tender.pdf" :
            st.subheader(f"Processing: {uploaded_file.name}")
            with st.spinner("Extracting tender details" , show_time = True) :
                extracted_text = extract_text_from_pdf(uploaded_file)
                tender_details = tender_details_gpt(extracted_text) 
                st.text_area("Etraxcted key details of tender" , value = tender_details , height = 300 , key = "td")
        training_data = pd.read_csv(cwd + "/models/al_hisotrical_data_feature_added.csv" , parse_dates=["Date"], index_col="Date") 
    with st.spinner("Forecasting Al price " , show_time = True) :
        # training_data = pd.read_csv(uploaded_file , parse_dates=["Date"], index_col="Date") 
        predicted_price = predict_price(
            date1="2024-08-29",
            vol1=10,
            price_change1=-1.62,
            open1=230.15,
            high1=230.15,
            low1=227.5,
            df= training_data
        )
    st.text_area("Aluminium Price prediction basde on tender" , value = f"\n📅 Predicted Price on 2024-08-29: {predicted_price}\n" , key = "alpp") 
    # import pdb; pdb.set_trace()
    with st.spinner("Fecthing last 7 day news" , show_time = True) : 
        news_df_7 = news_fetcher('tata project' , '7d')
    st.success("Fetched last 7 days news")
    with st.spinner("Fecthing last 1 month news" , show_time = True) : 
        news_df_40 = news_fetcher('tata project' , '40d')
    st.success("Fetched last 1 month news")
    with st.spinner("Fecthing last 1 year news" , show_time = True) : 
        news_df_365 = news_fetcher('tata project' , '365d')
    st.success("Fetched last 1 year news")
    with st.spinner("Calculating sentiment , rating and hightlights " , show_time = True) : 
        avg_sentiment_7 , rating_7 = analyze_sentiment(news_df_7)
        avg_sentiment_40 , rating_40 = analyze_sentiment(news_df_40)
        avg_sentiment_365 , rating_365 = analyze_sentiment(news_df_365)
    # print(avg_sentiment_7 , rating_7)

        big_news_7 = highlight_big_news(news_df_7)
        big_news_40 = highlight_big_news(news_df_40)
        big_news_365 = highlight_big_news(news_df_365)
    st.success("Sentiment analysis done !!")

    main_prompt = f"""
    You are a helpful financial assessment assistant for construction companies. You will be given a companies financial
    metrices , news and overall sentiment about company in last 7 , 40 and 365 days with an upcoming project details along with raw material prices. 
    Task1 : 
    Your task is to assess whether the company should be given the upcoming project or not based on company financials , news and sentiment rating.
    Financial details of company : {metrics_dict}
    Last one week sentiment about company : {avg_sentiment_7} 
    Last one week rating of company based on nwes : {rating_7}
    Last one week big news about company : {big_news_7}
    Last one month sentiment about company : {avg_sentiment_40} 
    Last one month rating of company based on nwes : {rating_40}
    Last one month big news about company : {big_news_40}
    Last one year sentiment about company : {avg_sentiment_365} 
    Last one year rating of company based on nwes : {rating_365}
    Last one year big news about company : {big_news_365}
    Tender details : {tender_details}
    Proposed budget by company : 110500
    Cement cost : 6000/MT

    Write your assesment in clear manner and rate whether this company should be given this project or not out of 10.
    Mention if there are any chances of delay or other diffculties in completing the project.
    Task2 :
    Provide out of 10 chances of the company going bankrupt by taking this project.
    Task3 :
    Output any big news that can impact decision of giving project to given company.
    Output the response in markdown format. Make sure to output final rating in large fonts.
    """

    def tender_details_gpt(main_prompt) : 
        resp = client.chat.completions.create(
            model = "gpt-4o-mini",
            temperature= 0,
            messages=[
                {
                    "role" : "system" , "content" : main_prompt
                },
                # {
                #     "role" : "user" , "content" : f"Here is the tender document : {text}"
                # }
            ] 
        )
        return resp.choices[0].message.content.strip()
    with st.spinner("Generating final assessment" , show_time = True) : 
        rate = tender_details_gpt(main_prompt) 
        st.markdown(rate)    
    # st.write(rate)









        # st.text_area("Extracted Text", extracted_text, height=150)
        # if st.button(f"Summarize {uploaded_file.name}"):
        #     summary = query_gpt(f"Summarize this document:\n{extracted_text}")
        #     st.text_area("AI Summary", summary, height=150)
