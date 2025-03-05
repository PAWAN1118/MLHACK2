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
        # Debugging: Print the JSON response to understand its structure
        st.write("API Response:", data)
        
        # Check if the response contains the expected data
        if 'data' in data and isinstance(data['data'], list):
            return pd.DataFrame(data['data'])
        else:
            st.error("Data format error. The API response does not contain the expected 'data' key or it is not a list.")
            return pd.DataFrame()
    else:
        st.error(f"Failed to fetch data. Status code: {r.status_code}")
        return pd.DataFrame()

def preprocess_data(df):
    # Ensure the DataFrame has enough columns for slicing
    if df.shape[1] < 55:
        st.error("The dataset does not have enough columns for preprocessing.")
        return pd.DataFrame()
    
    data2018 = df.iloc[:, 2:13].astype(float)
    data2019 = df.iloc[:, 13:24].astype(float)
    data2020 = df.iloc[:, 24:35].astype(float)
    data2021 = df.iloc[:, 35:45].astype(float)
    data2022 = df.iloc[:, 45:55].astype(float)
    
    all_years = pd.concat([data2018, data2019, data2020, data2021, data2022], axis=1)
    all_years.fillna(0, inplace=True)
    return all_years

st.title("Drug Cases Predictor App")
st.write("This app shows drug cases and predicts future cases using Machine Learning.")

year = st.selectbox("Select Year to View Data", ["2018", "2019", "2020", "2021", "2022"])

df = load_data()

if not df.empty:
    data = preprocess_data(df)
    
    if not data.empty:
        year_mapping = {
            "2018": data.iloc[:, :11],
            "2019": data.iloc[:, 11:22],
            "2020": data.iloc[:, 22:33],
            "2021": data.iloc[:, 33:43],
            "2022": data.iloc[:, 43:53],
        }
        
        selected_data = year_mapping.get(year)
        selected_data.columns = [f"Month {i+1}" for i in range(selected_data.shape[1])]

        st.subheader(f"Drug Cases in {year}")
        st.dataframe(selected_data)

        # Machine Learning
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.subheader("Prediction Results")
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R-Squared Score: {r2:.2f}")
        
        # Plot Graph
        st.subheader("Actual vs Predicted Cases")
        plt.figure(figsize=(10, 6))
        plt.plot(y_test.values, label="Actual Cases", color='blue')
        plt.plot(y_pred, label="Predicted Cases", color='red')
        plt.legend()
        plt.title("Actual vs Predicted Cases")
        plt.xlabel("Samples")
        plt.ylabel("Cases")
        st.pyplot(plt)

        # Future Prediction
        st.subheader("Predict Future Drug Cases")
        future_year = st.number_input("Enter the year for prediction", min_value=2023, max_value=2030, value=2023)
        future_month = st.number_input("Enter the month for prediction", min_value=1, max_value=12, value=1)

        if st.button("Predict Future Cases"):
            # Create a synthetic future dataset
            future_data = np.zeros((1, X.shape[1]))
            future_data[0, -1] = future_year  # Assuming the last column is the year
            future_data[0, -2] = future_month  # Assuming the second last column is the month

            # Predict future cases
            future_prediction = model.predict(future_data)
            st.write(f"Predicted Drug Cases for {future_month}/{future_year}: {future_prediction[0]:.2f}")

st.sidebar.write("Made with ❤️ by AI Student")
