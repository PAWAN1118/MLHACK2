import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the JSON data from the provided URL
url = "https://www.data.gov.in/backend/dms/v1/ogdp/resource/download/603189971/json/eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJkYXRhLmdvdi5pbiIsImF1ZCI6ImRhdGEuZ292LmluIiwiaWF0IjoxNzQxMTYxNzE3LCJleHAiOjE3NDExNjIwMTcsImRhdGEiOnsibmlkIjoiNjAzMTg5OTcxIn19.0G6wbxOJRrimBOB-OQmMx1rP8TcHXEZqgGGiGzBynqI"
response = requests.get(url)
data = response.json()

# Check if the request was successful
if response.status_code == 200:
    if "fields" in data and "data" in data:
        # Extract column names from 'fields' key
        columns = [field["label"] for field in data["fields"]]
        
        # Convert 'data' key into a DataFrame
        df = pd.DataFrame(data["data"], columns=columns)

        print("Data Preview:")
        print(df.head().to_string(index=False))
        
        # Separate yearly data into individual variables
        df_2018 = df.filter(like='2018')
        df_2019 = df.filter(like='2019')
        df_2020 = df.filter(like='2020')
        df_2021 = df.filter(like='2021')
        df_2022 = df.filter(like='2022')
        
        print("Data for 2018:")
        print(df_2018.head().to_string(index=False))
        print("Data for 2019:")
        print(df_2019.head().to_string(index=False))
        print("Data for 2020:")
        print(df_2020.head().to_string(index=False))
        print("Data for 2021:")
        print(df_2021.head().to_string(index=False))
        print("Data for 2022:")
        print(df_2022.head().to_string(index=False))
        
        # Reshape the dataset
        df_melted = df.melt(id_vars=["Sl. No.", "State/UT"], var_name="Year_DrugType", value_name="Seizure Quantity")
        df_melted[['Year', 'Drug Type']] = df_melted['Year_DrugType'].str.extract(r'(\d{4}) - (.+)')
        df_melted.drop(columns=['Year_DrugType'], inplace=True)
        df_melted.dropna(inplace=True)
        df_melted['Year'] = df_melted['Year'].astype(int)
        df_melted['Seizure Quantity'] = pd.to_numeric(df_melted['Seizure Quantity'], errors='coerce')
        df_melted.dropna(inplace=True)
        
        print("Reshaped Data Preview:")
        print(df_melted.head().to_string(index=False))
        
        # Splitting dataset into training and testing sets
        X = df_melted[['Year']]
        y = df_melted['Seizure Quantity']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train a Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Absolute Error: {mae}")
        print(f"Mean Squared Error: {mse}")
        
        # Plot Actual vs Predicted Values
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='dashed', label='Ideal Fit')
        plt.xlabel('Actual Seizure Quantity')
        plt.ylabel('Predicted Seizure Quantity')
        plt.title('Actual vs Predicted Drug Seizures')
        plt.legend()
        plt.show()
    else:
        print("Error: Unexpected JSON structure.")
else:
    print(f"Error: Failed to fetch data from the URL. Status code: {response.status_code}")
    print(f"Response content: {response.content}")
