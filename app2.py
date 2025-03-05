# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import streamlit as st
import requests
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the data from JSON
@st.cache_data  # Cache the data to improve performance
def load_data():
    try:
        # URL to the JSON data
        url = "https://www.data.gov.in/backend/dms/v1/ogdp/resource/download/603189971/json/eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJkYXRhLmdvdi5pbiIsImF1ZCI6ImRhdGEuZ292LmluIiwiaWF0IjoxNzQxMDYzNTIyLCJleHAiOjE3NDEwNjM4MjIsImRhdGEiOnsibmlkIjoiNjAzMTg5OTcxIn19.ikJQgi0GtPlW6WBQs2WshuBHNG18w6Mk04KTY3Nrv0s"
        
        # Fetch the JSON data from the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()

        # Check if the JSON structure is as expected
        if 'data' not in data:
            st.error("The JSON data does not contain the expected 'data' key.")
            return None

        # Convert JSON to DataFrame
        df = pd.DataFrame(data['data'])

        # Fill missing values if any
        df.fillna(0, inplace=True)

        # Convert columns to appropriate data types if necessary
        df = df.apply(pd.to_numeric, errors='coerce')

        # Drop any rows with missing values if necessary
        df.dropna(inplace=True)

        return df

    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None

# Load the data
df = load_data()

if df is not None:
    # Step 2: Feature selection and target variable
    # Assuming the target variable is the last column (e.g., '2022 - Opi')
    # and the features are all other columns
    features = df.columns[:-1]  # All columns except the last one
    target = df.columns[-1]     # The last column is the target

    X = df[features]
    y = df[target]

    # Step 3: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Train a machine learning model
    @st.cache_resource  # Cache the model to avoid retraining
    def train_model():
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    model = train_model()

    # Step 5: Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Step 6: Save the model (optional)
    joblib.dump(model, 'opioid_cases_predictor.pkl')

    # Step 7: Streamlit App
    st.title("Opioid Cases Predictor")
    st.write("This app predicts future opioid cases based on historical data.")

    # Display the dataset
    st.subheader("Dataset")
    st.write(df)

    # Plot the dataset (e.g., distribution of the target variable)
    st.subheader("Distribution of Target Variable")
    fig, ax = plt.subplots()
    ax.hist(y, bins=20, color='blue', alpha=0.7)
    ax.set_xlabel(target)
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Target Variable")
    st.pyplot(fig)

    # Display model evaluation metrics
    st.subheader("Model Evaluation")
    st.write(f"Mean Squared Error (MSE): {mse}")

    # Plot actual vs predicted values
    st.subheader("Actual vs Predicted Values")
    fig2, ax2 = plt.subplots()
    ax2.scatter(y_test, y_pred, color='green', alpha=0.5)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax2.set_xlabel("Actual Values")
    ax2.set_ylabel("Predicted Values")
    ax2.set_title("Actual vs Predicted Values")
    st.pyplot(fig2)

    # Input fields for user to enter feature values
    st.subheader("Enter Feature Values for Prediction")
    input_data = {}
    for feature in features:
        input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

    # Convert input data into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure the columns match the training data
    input_df = input_df.reindex(columns=features, fill_value=0)

    # Make prediction
    if st.button("Predict"):
        prediction = model.predict(input_df)
        st.success(f"Predicted Future Opioid Cases: {prediction[0]}")

        # Plot the prediction
        st.subheader("Prediction Visualization")
        fig3, ax3 = plt.subplots()
        ax3.bar(["Predicted Value"], [prediction[0]], color='orange', alpha=0.7)
        ax3.set_ylabel("Opioid Cases")
        ax3.set_title("Predicted Future Opioid Cases")
        st.pyplot(fig3)
