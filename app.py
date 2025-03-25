# Core Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from groq import Groq
import os
from dotenv import load_dotenv

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Streamlit UI Styling
st.set_page_config(page_title="📈 AI Forecasting Agent", page_icon="🔮", layout="wide")
st.title("🔮 AI Forecasting Agent using Prophet")

# Upload Excel File
uploaded_file = st.file_uploader("📤 Upload Excel File with `Date` and `Revenue` Columns", type=["xlsx"])
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        if 'Date' not in df.columns or 'Revenue' not in df.columns:
            st.error("❌ Your file must contain 'Date' and 'Revenue' columns.")
        else:
            # Preprocess data for Prophet
            df = df[['Date', 'Revenue']].rename(columns={"Date": "ds", "Revenue": "y"})
            df['ds'] = pd.to_datetime(df['ds'])
            df = df.sort_values('ds')

            # Display raw data
            st.subheader("📊 Raw Time Series Data")
            st.line_chart(df.set_index('ds')['y'])

            # Forecasting
            st.subheader("📈 Forecasting Future Revenue")
            periods = st.slider("⏱ Forecast Horizon (Days)", min_value=7, max_value=365, value=90)
            model = Prophet()
            model.fit(df)

            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)

            # Plot Forecast
            fig1 = model.plot(forecast)
            st.pyplot(fig1)

            # Plot Components
            st.subheader("📉 Forecast Components")
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)

            # Optional: FP&A-style AI commentary
            with st.expander("🤖 Generate AI Insights (Optional)"):
                client = Groq(api_key=GROQ_API_KEY)
                data_for_ai = df.to_json(orient='records')
                prompt = f"""
                You are an AI financial analyst. Provide:
                - Key trends and anomalies in revenue
                - Possible seasonality or growth patterns
                - Recommendations based on the forecast

                Here is the time series data:
                {data_for_ai}
                """
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are an expert time series forecaster."},
                        {"role": "user", "content": prompt}
                    ],
                    model="llama3-8b-8192"
                )
                ai_commentary = response.choices[0].message.content
                st.subheader("🧠 AI-Generated Forecast Commentary")
                st.write(ai_commentary)

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
