import streamlit as st
import os
from dotenv import load_dotenv 
from pypdf import PdfReader 
from openai import OpenAI 
import json
import pandas as pd

# from st_app import extract_text_from_pdf , query_gpt , tender_details_gpt , client
from custom_functions import predict_bankruptcy_selected_feature , predict_price 
from custom_functions import news_fetcher , analyze_sentiment , highlight_big_news , tender_details_gpt , extract_text_from_pdf , query_gpt_selected_feature , tender_details_gpt , query_gpt_xgb

cwd = os.getcwd() 


# Load environment variables
load_dotenv() 
client = OpenAI(api_key= os.getenv('OPENAI_API_KEY')) 



st.title("Construction Vendor Bankruptcy Analysis")

uploaded_files = st.file_uploader('''Upload 2 pdf files simultaneously - \n 1. company_finanicials.pdf : Document containing financial report of company 
2. tender_details.pdf : pdf document contaning tender details''', type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:

        # st.subheader(f"Processing: {uploaded_file.name}") 
        # import pdb; pdb.set_trace()
        if "financials" in uploaded_file.name :
            st.subheader(f"Processing: {uploaded_file.name}")
            company_name = uploaded_file.name.split("_financials")[0]
            company_name = company_name.replace("_" , " ")

            with st.spinner("Accessing bankruptcy" , show_time = True) :
                # st.subheader(f"Processing: {uploaded_file.name}")
                # import pdb; pdb.set_trace()
                extracted_text = extract_text_from_pdf(uploaded_file)
                print(extracted_text)
                print("++++++++++++++++++++++++")
                # metrics_dict = json.loads(query_gpt_selected_feature(extracted_text)) 
                try :
                    metrics_dict = json.loads(query_gpt_xgb(extracted_text)) 
                    print(metrics_dict)
                    bankruptcy_status = predict_bankruptcy_selected_feature(metrics_dict)
                    print(bankruptcy_status)
                    if bankruptcy_status[0] == 1 : 
                        status = "Not bankrupt yet" 
                        st.text_area("Bankruptcy status prediction" ,value = f"{status}" , key = "bsp0")
                    if bankruptcy_status[0] == 0 : 
                        status = "About to be bankrupt"
                        st.text_area("Bankruptcy status prediction" ,value = f"{status}", key = "bsp1")
                except Exception as e:
                    print("error :" , e)
                    status = "Could not be determined due to incomplete information"
                    st.text_area("Bankruptcy status prediction" ,value = f"""Incomplete financial details to predict bankruptcy. Reuired metrices - 
                    "current assests" : float ,
                    "cost of good sold" : float , 
                    "Depreciation and amortization" : float ,
                    "EBITDA" : float ,
                    "Inventory" : float,
                    "Net Income" : float ,
                    "Total Receivables" : float , 
                    "Market value" : float , 
                    "Net sales" : float , 
                    "Total assests" : float , 
                    "Total Long-term debt " : float ,
                    "EBIT" : float , 
                    "Gross Profit" : float , 
                    "Total Current Liabilities" : float , 
                    "Retained Earnings" : float , 
                    "Total Revenue" : float , 
                    "Total Liabilities " : float , 
                    "Total Operating Expenses" : float""", height = 300 , key = "bsp1")

        if "tender" in uploaded_file.name  :
            st.subheader(f"Processing: {uploaded_file.name}")
            with st.spinner("Extracting tender details" , show_time = True) :
                extracted_text = extract_text_from_pdf(uploaded_file)
                tender_details = tender_details_gpt(extracted_text) 
                st.text_area("Etraxcted key details of tender" , value = tender_details , height = 300 , key = "td")
        training_data = pd.read_csv(cwd + "/models/al_hisotrical_data_feature_added.csv" , parse_dates=["Date"], index_col="Date") 
    with st.spinner("Forecasting Al price " , show_time = True) :
        # training_data = pd.read_csv(uploaded_file , parse_dates=["Date"], index_col="Date")
        # import pdb; pdb.set_trace()
        tender_details_dict = json.loads(tender_details) 
        if tender_details_dict["Project starting date"] != "NA" :
            actual_date = tender_details_dict["Project starting date"]
            date1 = tender_details_dict["Project starting date"]
        else :
            date1="2024-08-29"
        predicted_price = predict_price(
            date1= date1,
            vol1=10,
            price_change1=-1.62,
            open1=230.15,
            high1=230.15,
            low1=227.5,
            df= training_data
        )
    # st.text_area("Aluminium Price prediction basde on tender" , value = f"\n📅 Predicted Price on {actual_date}: {predicted_price}\n" , key = "alpp") 
    # import pdb; pdb.set_trace()
    with st.spinner("Fecthing last 7 day news" , show_time = True) : 
        news_df_7 = news_fetcher(company_name , '7d')
    st.success("Fetched last 7 days news")
    with st.spinner("Fecthing last 1 month news" , show_time = True) : 
        news_df_40 = news_fetcher(company_name , '40d')
    st.success("Fetched last 1 month news")
    with st.spinner("Fecthing last 1 year news" , show_time = True) : 
        news_df_365 = news_fetcher(company_name , '365d')
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

    # main_prompt = f"""
    # You are a helpful financial assessment assistant for construction companies. You will be given a companies financial
    # metrices , news and overall sentiment about company in last 7 , 40 and 365 days with an upcoming project details along with raw material prices. 
    # Task1 : 
    # Your task is to assess whether the company should be given the upcoming project or not based on company financials , news and sentiment rating.
    # Financial details of company : {metrics_dict}
    # Bankruptcy status of company : {status} 
    # Last one week sentiment about company : {avg_sentiment_7} 
    # Last one week rating of company based on nwes : {rating_7}
    # Last one week big news about company : {big_news_7}
    # Last one month sentiment about company : {avg_sentiment_40} 
    # Last one month rating of company based on nwes : {rating_40}
    # Last one month big news about company : {big_news_40}
    # Last one year sentiment about company : {avg_sentiment_365} 
    # Last one year rating of company based on nwes : {rating_365}
    # Last one year big news about company : {big_news_365}
    # Tender details : {tender_details}
    # Cement cost : $150/MT
    # steel cost : ${predicted_price}/MT
    # Actual project cost : to be calculated from raw material quantiy from tender details and cement and steel prices. Keep margin for operation costs too while calculating actual cost of project.
    # Write your assesment in clear manner and rate whether this company should be given this project or not out of 10.
    # Mention if there are any chances of delay or other diffculties in completing the project.
    # Task2 :
    # Provide out of 10 chances of the company going bankrupt by taking this project.
    # Task3 :
    # Output any big news that can impact decision of giving project to given company.
    # Output the response in markdown format. Make sure to output final rating in large fonts.
    # """
    main_prompt = f"""
You are a financial assessment assistant specializing in evaluating construction companies for project eligibility.

## Input Data:
You will receive the following details about a company:
- Financial Metrics: {metrics_dict}
- Bankruptcy Status: {status}
- Sentiment Analysis:
  - Last 7 Days: Sentiment Score: {avg_sentiment_7}, Rating: {rating_7}, Major News: {big_news_7}
  - Last 40 Days: Sentiment Score: {avg_sentiment_40}, Rating: {rating_40}, Major News: {big_news_40}
  - Last 365 Days: Sentiment Score: {avg_sentiment_365}, Rating: {rating_365}, Major News: {big_news_365}
- Tender Details: {tender_details}
- Raw Material Prices:
  - Cement: $150/MT
  - Steel: ${predicted_price}/MT

## Project Cost Calculation:
The actual project cost should be estimated using the raw material quantities mentioned in the tender details, while also accounting for operational and overhead costs by applying a margin.

1. Calculate Raw Material Cost:
   - Extract the required quantities of cement and steel from tender details.
   - Compute total raw material cost:
     - `Total Cement Cost = Cement Quantity * $150/MT`
     - `Total Steel Cost = Steel Quantity * ${predicted_price}/MT`
   - Sum these to get the base material cost.

2. Apply Operational Cost Margin:
   - Since project costs include more than just raw materials, apply a margin of X% (default: 10-30%) to account for labor, logistics, and other operational costs.
   - `Actual Project Cost = Base Material Cost * (1 + Margin)`

The margin percentage should be chosen based on industry norms and company-specific conditions.

---

### Task 1: Project Eligibility Assessment
Assess whether the company should be awarded the upcoming project based on:
- Financial health based on Financial Metrices and Bankruptcy status.
- Unavailability of bankrutcy status must be treated as warning sign for elegibility assessment.
- Sentiment analysis
- News impact
- Ability to cover project costs
- Tender details and actual project costs

Output Requirements:
- Clearly explain the assessment.
- Provide a rating out of 10 on whether the company should receive the project.
- Identify potential delays or risks that could affect project completion.

---

### Task 2: Bankruptcy Risk Evaluation
Estimate the likelihood (out of 10) of the company going bankrupt if awarded this project.

---

### Task 3: Critical News Impact
Highlight any significant news that could influence the decision to award the project.

---

### Output Format:
- Response should be in Markdown format.
- The final project rating must be displayed in large font.
"""



    def tender_details_gpt(main_prompt) : 
        resp = client.chat.completions.create(
            model = "gpt-4o-mini",
            # model = "gpt-4.5-preview",
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

