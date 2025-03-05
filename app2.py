import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Load the JSON data from the provided URL
url ="https://www.data.gov.in/backend/dms/v1/ogdp/resource/download/603189971/json/eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJkYXRhLmdvdi5pbiIsImF1ZCI6ImRhdGEuZ292LmluIiwiaWF0IjoxNzQxMTYxNzE3LCJleHAiOjE3NDExNjIwMTcsImRhdGEiOnsibmlkIjoiNjAzMTg5OTcxIn19.0G6wbxOJRrimBOB-OQmMx1rP8TcHXEZqgGGiGzBynqI"

# Check if the request was successful
if response.status_code == 200:
    try:
        data = response.json()
        if "fields" in data and "data" in data:
            # Extract column names from 'fields' key
            columns = [field["label"] for field in data["fields"]]
            
            # Convert 'data' key into a DataFrame
            df = pd.DataFrame(data["data"], columns=columns)
            
            # Reshape the dataset
            df_melted = df.melt(id_vars=["Sl. No.", "State/UT"], var_name="Year_DrugType", value_name="Seizure Quantity")
            df_melted[['Year', 'Drug Type']] = df_melted['Year_DrugType'].str.extract(r'(\d{4}) - (.+)')
            df_melted.drop(columns=['Year_DrugType'], inplace=True)
            df_melted.dropna(inplace=True)
            df_melted['Year'] = df_melted['Year'].astype(int)
            df_melted['Seizure Quantity'] = pd.to_numeric(df_melted['Seizure Quantity'], errors='coerce')
            df_melted.dropna(inplace=True)
            
            # Streamlit App
            st.title("NDPS Seizure Analysis")
            year_input = st.number_input("Enter Year (2018-2022):", min_value=2018, max_value=2022, step=1)
            
            # Filter data based on the selected year
            filtered_df = df_melted[df_melted['Year'] == year_input]
            
            if not filtered_df.empty:
                st.write(f"Seizure Data for the Year {year_input}")
                st.dataframe(filtered_df)
                
                # Plot the graph
                st.write("### Seizure Quantity by Drug Type")
                fig, ax = plt.subplots(figsize=(10, 5))
                filtered_df.groupby("Drug Type")['Seizure Quantity'].sum().plot(kind='bar', ax=ax, color='skyblue')
                ax.set_xlabel("Drug Type")
                ax.set_ylabel("Seizure Quantity")
                ax.set_title(f"Seizure Quantity by Drug Type in {year_input}")
                st.pyplot(fig)
            else:
                st.write("No data available for the selected year.")
        else:
            st.write("Error: Unexpected JSON structure.")
    except ValueError:
        st.write("Error: Failed to parse JSON response.")
else:
    st.write(f"Error: Failed to fetch data from the URL. Status code: {response.status_code}")
