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
    data2018 = df.iloc[:, 2:13].replace('-', np.nan).astype(float)
    data2019 = df.iloc[:, 13:24].replace('-', np.nan).astype(float)
    data2020 = df.iloc[:, 24:35].replace('-', np.nan).astype(float)
    data2021 = df.iloc[:, 35:45].replace('-', np.nan).astype(float)
    data2022 = df.iloc[:, 45:55].replace('-', np.nan).astype(float)

    all_years = pd.concat([data2018, data2019, data2020, data2021, data2022], axis=1)
    all_years.fillna(0, inplace=True)
    return all_years

st.title("Drug Cases Predictor App 🚨")
st.write("This app shows drug cases and predicts future cases using Machine Learning.")

year = st.selectbox("Select Year to View Data", ["2018", "2019", "2020", "2021", "2022"])

df = load_data()

if not df.empty:
    data = preprocess_data(df)

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

    # Machine Learning Model
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

    # Plotting Graph
    st.subheader("Actual vs Predicted Cases")
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label="Actual Cases", color='blue', marker='o')
    plt.plot(y_pred, label="Predicted Cases", color='red', linestyle='dashed', marker='x')
    plt.title("Actual vs Predicted Cases")
    plt.xlabel("Samples")
    plt.ylabel("Cases")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

    # Future Year Predictions
    st.sidebar.title("Future Predictions 🔮")
    future_year = st.sidebar.slider("Predict Future Year", 2023, 2030, 2023)
    latest_data = X.iloc[-1].values.reshape(1, -1)
    future_pred = model.predict(latest_data)[0]
    st.sidebar.subheader(f"Predicted Drug Cases in {future_year}: {int(future_pred)}")

st.sidebar.write("Made with ❤️ by AI Student")
