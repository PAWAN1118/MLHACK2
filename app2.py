import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

url = "https://www.data.gov.in/backend/dms/v1/ogdp/resource/download/603189971/json/eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJkYXRhLmdvdi5pbiIsImF1ZCI6ImRhdGEuZ292LmluIiwiaWF0IjoxNzQxMDYzNTIyLCJleHAiOjE3NDEwNjM4MjIsImRhdGEiOnsibmlkIjoiNjAzMTg5OTcxIn19.ikJQgi0GtPlW6WBQs2WshuBHNG18w6Mk04KTY3Nrv0s"

@st.cache_data
def load_data():
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        if 'data' in data and isinstance(data['data'], list):
            return pd.DataFrame(data['data'])
        else:
            st.error("Data format error.")
            return pd.DataFrame()
    else:
        st.error("Failed to fetch data.")
        return pd.DataFrame()

def preprocess_data(df):
    # Replace hyphens or other non-numeric characters with NaN
    df = df.replace('-', np.nan)  # Replace '-' with NaN
    # You might need to replace other non-numeric characters as well
    
    data2018 = df.iloc[:, 2:13].astype(float)
    data2019 = df.iloc[:, 13:24].astype(float)
    data2020 = df.iloc[:, 24:35].astype(float)
    data2021 = df.iloc[:, 35:45].astype(float)
    data2022 = df.iloc[:, 45:55].astype(float)
    
    all_years = pd.concat([data2018, data2019, data2020, data2021, data2022], axis=1)
    all_years.fillna(0, inplace=True)  # Fill NaN values with 0
    return all_years
st.title("Drug Cases Prediction App")
st.write("This app shows drug cases and predicts future cases using Machine Learning.")

year = st.selectbox("Select Year to View Data", ["2018", "2019", "2020", "2021", "2022"])
years_to_predict = st.slider("Select Future Years to Predict", 1, 10, 3)

# Load Data
df = load_data()

if not df.empty:
    data = preprocess_data(df)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.subheader("Model Evaluation")
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R-Squared Score: {r2:.2f}")
    
    # Future Predictions
    future_years = [2023 + i for i in range(years_to_predict)]
    future_predictions = model.predict(np.tile(X.iloc[-1].values, (years_to_predict, 1)))
    future_df = pd.DataFrame({"Year": future_years, "Predicted Cases": future_predictions})
    
    st.subheader("Future Predictions")
    st.dataframe(future_df)
    
    # Plot Predictions
    st.subheader("Prediction Graph")
    plt.figure(figsize=(10, 6))
    plt.plot(future_df["Year"], future_df["Predicted Cases"], color='green', marker='o')
    plt.title("Predicted Cases for Future Years")
    plt.xlabel("Year")
    plt.ylabel("Cases")
    plt.grid(True)
    st.pyplot(plt)

st.sidebar.write("Made with ❤️ by AI Student")
