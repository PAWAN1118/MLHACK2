import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit App
st.title("Drug Cases Prediction App")
st.write("This app shows drug cases and predicts future cases using Machine Learning.")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

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

st.sidebar.markdown("Made By DEEPTHINKERS")
