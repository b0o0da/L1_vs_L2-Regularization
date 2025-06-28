import streamlit as st
import pickle
import numpy as np
import pandas as pd 
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("ad.csv")
X = df.drop(columns="Sales")
y = df["Sales"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Load saved model, poly transformer, and feature list
model_path = "model.pkl"  
with open(model_path, "rb") as file:
    saved = pickle.load(file)
    model = saved["model"]
    poly = saved["poly"]
    features = saved["features"]

# Streamlit UI
st.title("📺📻🗞️ Sales Prediction App")
st.write("Enter advertising budget values below:")

# Inputs for TV, Radio, Newspaper
tv = st.number_input("📺 TV Budget ($)", min_value=0.0, max_value=1000.0, step=1.0, value=0.0)
radio = st.number_input("📻 Radio Budget ($)", min_value=0.0, max_value=1000.0, step=1.0, value=0.0)
newspaper = st.number_input("🗞️ Newspaper Budget ($)", min_value=0.0, max_value=1000.0, step=1.0, value=0.0)

# Predict button
if st.button("🔮 Predict Sales"):
    # Prepare input using the saved features
    input_data = pd.DataFrame([[tv, radio, newspaper]], columns=features)
    input_poly = poly.transform(input_data)
    
    # Predict
    predicted_sales = model.predict(input_poly)[0]
    st.success(f"🛒 Predicted Sales: **{predicted_sales:.2f} units**")

    # Evaluate model on test data
    X_test_poly = poly.transform(X_test[features])
    y_pred = model.predict(X_test_poly)
    mae = mean_absolute_error(y_test, y_pred)
    st.info(f"📊 MAE on test data: **{mae:.2f} units**")
