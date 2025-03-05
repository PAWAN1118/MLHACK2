import streamlit as st
import pandas as pd
import numpy as np
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
    df = pd.read_csv(url)
    return df

# Data Loading
df = load_data()

# Preprocess data
data2018 = df.iloc[:, 2:13].astype(float)
data2019 = df.iloc[:, 13:24].astype(float)
data2020 = df.iloc[:, 24:35].astype(float)
data2021 = df.iloc[:, 35:45].astype(float)
data2022 = df.iloc[:, 45:55].astype(float)

# Concatenate all years
data_all = pd.concat([data2018, data2019, data2020, data2021, data2022], axis=1)
data_all.fillna(0, inplace=True)

# Create features and target
X = data_all.iloc[:, :-1]
y = data_all.iloc[:, -1]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Year selection
selected_year = st.selectbox("Select Year to View Data", [2018, 2019, 2020, 2021, 2022])
future_years = st.slider("Select Future Years to Predict", 1, 10, 3)

if st.button("Predict"):
    if selected_year == 2022:
        future_data = data2022.copy()
        predictions = []
        for year in range(future_years):
            pred = model.predict([future_data.values[0]])
            predictions.append(pred[0])
            future_data = np.roll(future_data, -1)
            future_data[-1] = pred[0]
        st.write(f"Predicted Cases for next {future_years} years:", predictions)
    else:
        st.error("Prediction is only available from 2022 onward.")

st.sidebar.markdown("Made with ❤️ by AI Student")
