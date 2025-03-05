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
    # URL to the JSON data
    url = "https://www.data.gov.in/backend/dms/v1/ogdp/resource/download/603189971/json/eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJkYXRhLmdvdi5pbiIsImF1ZCI6ImRhdGEuZ292LmluIiwiaWF0IjoxNzQxMDYzNTIyLCJleHAiOjE3NDEwNjM4MjIsImRhdGEiOnsibmlkIjoiNjAzMTg5OTcxIn19.ikJQgi0GtPlW6WBQs2WshuBHNG18w6Mk04KTY3Nrv0s"
    
    # Fetch the JSON data from the URL
    response = requests.get(url)
    data = response.json()

    # Convert JSON to DataFrame
    df = pd.DataFrame(data['data'])
    
    # Print the shape of the DataFrame after loading from JSON
    print("Shape of DataFrame after loading from JSON:", df.shape) 
    
    # Fill missing values if any
    # Instead of filling with 0, consider filling with the mean or median
    # or investigate why there are missing values and potentially drop the columns 
    # with a lot of missing data instead of the entire row.
    # df.fillna(df.mean(), inplace=True) # Example: Filling with the mean
    
    # Print the shape of the DataFrame after filling missing values
    print("Shape of DataFrame after filling missing values:", df.shape)

    # Convert columns to appropriate data types if necessary
    # Investigating the data types and errors='coerce' would be important
    # errors='coerce' will replace invalid values with NaN.
    # Ensure data type conversion works as expected and doesn't lead to excessive data loss
    # For instance, print dtypes before and after or check for specific problematic conversions.
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Print the shape of the DataFrame after converting to numeric
    print("Shape of DataFrame after converting to numeric:", df.shape)

    # Drop any rows with missing values if necessary
    # Before dropping rows, carefully evaluate how many rows are being dropped and why
    # Potentially drop only specific columns instead or try to impute the missing values.
    # df.dropna(inplace=True)  # Commenting out or changing based on why it is empty
    # Check the impact of the imputation or alternative strategies before proceeding

    # Print the shape of the DataFrame after dropping rows with missing values
    print("Shape of DataFrame after dropping rows with missing values:", df.shape)

    return df


# Load the data
df = load_data()

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
