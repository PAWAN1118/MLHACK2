import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit App
st.title("Drug Cases Prediction App")
st.write("This app shows drug cases and predicts future cases using Machine Learning.")

# Load Data
@st.cache_data
def load_data():
    url = "https://www.data.gov.in/backend/dms/v1/ogdp/resource/download/603189971/json/eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJkYXRhLmdvdi5pbiIsImF1ZCI6ImRhdGEuZ292LmluIiwiaWF0IjoxNzQxMDYzNTIyLCJleHAiOjE3NDEwNjM4MjIsImRhdGEiOnsibmlkIjoiNjAzMTg5OTcxIn19.ikJQgi0GtPlW6WBQs2WshuBHNG18w6Mk04KTY3Nrv0s"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'data' in data and isinstance(data['data'], list):
            df = pd.DataFrame(data['data'])
            return df
    return pd.DataFrame()

# Data Loading
df = load_data()

# Check if data is empty
if df.empty:
    st.error("Failed to load data.")
else:
    # Preprocess data
    data_years = [df.iloc[:, 2:13], df.iloc[:, 13:24], df.iloc[:, 24:35], df.iloc[:, 35:45], df.iloc[:, 45:55]]
    data_all = pd.concat([year.astype(float) for year in data_years], axis=1)
    data_all.fillna(0, inplace=True)

    # Create features and target
    X = data_all.iloc[:, :-1]
    y = data_all.iloc[:, -1] if data_all.shape[1] > 0 else pd.Series(np.zeros(data_all.shape[0]))

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Year input
    selected_year = st.number_input("Enter Year to Predict Cases", min_value=2023, max_value=2035, step=1)

    if st.button("Predict"):
        future_data = data_years[-1].mean().values
        years_to_predict = selected_year - 2022
        predictions = []
        for year in range(years_to_predict):
            pred = model.predict([future_data])
            predictions.append(pred[0])
            future_data = np.roll(future_data, -1)
            future_data[-1] = pred[0]
        st.write(f"Predicted Cases for {selected_year}: {predictions[-1]}")

st.sidebar.markdown("Made with ❤️ by AI Student")
