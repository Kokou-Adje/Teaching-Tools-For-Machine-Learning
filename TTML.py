# TTML.py
"""
Kokou Adje
CS 7993 AI Capstone - W01 – Spring 2025
[AC-5 Red] Teaching Tools for Machine Learning
"""

# Import libraries
import streamlit as strmlit
import pandas as panda
import numpy as nump
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as matplt


# Dataset uploading and loading
def upload_dataset():
    uploaded_data = strmlit.file_uploader("Upload weather data", type="csv")
    try:
        # Verify if data is uploaded
        if uploaded_data is not None:
            panda_data = panda.read_csv(uploaded_data)
        else:
            # When no data uploaded use local data
            strmlit.warning("Local dataset (weather_data.csv) is using. No file selected")
            panda_data = panda.read_csv('weather_data.csv')
        return panda_data
    except FileNotFoundError:
        strmlit.error(
            "No Local dataset (weather_data.csv) found. Please upload a file for visualization")
        return None


# Dataset structure validation
def validate_dataset(panda_data):
    columns_required = ['Humidity', 'Visibility (km)', 'Temperature (C)']
    if not all(col in panda_data.columns for col in columns_required):
        strmlit.error(f"The dataset requires the following columns: {', '.join(columns_required)}")
        strmlit.strmlitop()


# Preview data
def preview_data(panda_data):
    if strmlit.checkbox("Preview data"):
        strmlit.subheader("Dataset")
        strmlit.write(panda_data.head())


# Function to train the linear regression model
def train_lin_reg_model(panda_data):
    features = panda_data[['Humidity', 'Visibility (km)']]
    label = panda_data['Temperature (C)']
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(features, label)
    return lin_reg_model


# Predict using user inputs
def predict(lin_reg_model):
    strmlit.subheader("Predict Temperature")
    humidity_column, visibility_column = strmlit.columns(2)
    with humidity_column:
        humid = strmlit.number_input("Humidity (0.0-1.0):", min_value=0.0, max_value=1.0, value=0.5)
    with visibility_column:
        visibil = strmlit.number_input("Visibility (km):", min_value=0.0, max_value=1.0, value=0.5)

    if strmlit.button("Predict Temperature"):
        features = [[humid, visibil]]
        temp_prediction = lin_reg_model.predict(features)[0]
        strmlit.success(f"Temperature Predicted: {temp_prediction:.2f}°C")


# Visualization of the regression surface
def lin_reg_model_visualization(panda_data, lin_reg_model):
    strmlit.subheader("Model Visualization")

    # Creation of meshgrid for surface plot
    humidity_interval = nump.linspace(panda_data['Humidity'].min(), panda_data['Humidity'].max(), 20)
    visibility_interval = nump.linspace(panda_data['Visibility (km)'].min(), panda_data['Visibility (km)'].max(), 20)
    humid, visib = nump.meshgrid(humidity_interval, visibility_interval)
    temp_pred = lin_reg_model.predict(nump.c_[humid.ravel(), visib.ravel()]).reshape(humid.shape)

    # Regression surface visualization
    figure = matplt.figure(figsize=(10, 6))
    axes = figure.add_subplot(111, projection='3d')
    axes.scatter(panda_data['Humidity'], panda_data['Visibility (km)'], panda_data['Temperature (C)'], c='red', label='Actual Data')
    axes.plot_surface(humid, visib, temp_pred, alpha=0.5, cmap='viridis')

    axes.set_xlabel('Humidity')
    axes.set_ylabel('Visibility (km)')
    axes.set_zlabel('Temperature (°C)')
    axes.set_title('Linear Regression Surface')
    axes.legend()

    strmlit.pyplot(figure)


# Main function
def ttml_main():
    strmlit.title("Teaching Tools for Machine Learning")
    strmlit.write("TTML is a web-based application implemented to explore some machine learning algorithms. Users can "
                  "experiment supervised and unsupervised learning algorithms through visualizations. For this first "
                  "version, we will focus on linear regression.")
    strmlit.write("This web application will predict temperature using both humidity and visibility with Linear "
                  "Regression and dataset visualization")

    # Dataset uploading / Loading
    panda_data = upload_dataset()
    if panda_data is not None:
        validate_dataset(panda_data)
        preview_data(panda_data)

        # Model training
        lin_reg_model = train_lin_reg_model(panda_data)

        # Prediction making
        predict(lin_reg_model)

        # Model visualization
        lin_reg_model_visualization(panda_data, lin_reg_model)


# Run the app
if __name__ == "__main__":
    ttml_main()
