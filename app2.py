import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# API URL
url = "https://www.data.gov.in/backend/dms/v1/ogdp/resource/download/603189971/json/eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJkYXRhLmdvdi5pbiIsImF1ZCI6ImRhdGEuZ292LmluIiwiaWF0IjoxNzQxMDYzNTIyLCJleHAiOjE3NDEwNjM4MjIsImRhdGEiOnsibmlkIjoiNjAzMTg5OTcxIn19.ikJQgi0GtPlW6WBQs2WshuBHNG18w6Mk04KTY3Nrv0s"

@st.cache_data
def load_data():
    """Fetch data from API and return DataFrame"""
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        st.write("API Response Sample:", data)  # Debug: Show API response
        if 'data' in data and isinstance(data['data'], list):
            return pd.DataFrame(data['data'])
        else:
            st.error("Data format error. Available keys: " + str(data.keys()))
            return pd.DataFrame()
    else:
        st.error("Failed to fetch data. Status Code: " + str(r.status_code))
        return pd.DataFrame()

def preprocess_data(df):
    """Process raw data and handle missing values"""
    years = [2018, 2019, 2020, 2021, 2022]
    data_dict = {}
    
    for i, year in enumerate(years):
        start, end = 2 + (i * 11), 2 + ((i + 1) * 11)
        if end > df.shape[1]:
            st.error(f"Year {year} data out of range. Columns available: {df.shape[1]}")
            continue
        temp_df = df.iloc[:, start:end].replace('-', np.nan).astype(float)
        temp_df.fillna(temp_df.median(), inplace=True)
        data_dict[year] = temp_df
    
    return data_dict

# Load & preprocess data
df = load_data()
if not df.empty:
    data = preprocess_data(df)

    st.sidebar.title("Controls")
    year = st.sidebar.selectbox("Select Year to View Data", list(data.keys()))
    selected_data = data[year]
    
    st.title("Drug Cases Data Viewer")
    st.subheader(f"Drug Cases in {year}")
    st.dataframe(selected_data)
    
    # Visualization using Matplotlib
    st.subheader(f"Visualization of Drug Cases in {year}")
    plt.figure(figsize=(10, 5))
    plt.plot(selected_data.mean(), marker='o', linestyle='-', color='b')
    plt.xlabel("Months")
    plt.ylabel("Cases")
    plt.title(f"Drug Cases Trend in {year}")
    plt.grid(True)
    st.pyplot(plt)

st.sidebar.write("Made with ❤️ by AI Student")
