import streamlit as st 
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge,Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
feature_names = diabetes.feature_names

def load_ridge_model():
    with open("ridge_model.pkl", "rb") as file:
        ridge_model, scaler = pickle.load(file)
    return ridge_model, scaler

def load_lasso_model():
    with open("lasso_model.pkl", "rb") as file:
        lasso_model, scaler = pickle.load(file)
    return lasso_model, scaler

ridge_model, scaler = load_ridge_model()
lasso_model, _ = load_lasso_model()


st.title("Diabetes Prediction using Ridge and Lasso Regression")
st.subheader("Enter Patient Data")

input_features = []
for feature in feature_names:
      value = st.slider(
              f"{feature}",
              -2.0,2.0,0.0,0.01
      )
      input_features.append(value)

input_data = np.array(input_features).reshape(1,-1)

ridge_prediction = ridge_model.predict(input_data)[0]
lasso_prediction = lasso_model.predict(input_data)[0]

st.subheader("Predicted Diabetes Progression Score")
st.write(f" ***Ridge Regression Prediction:*** {ridge_prediction:.2f}")
st.write(f" ***Lasso Regression Prediction:*** {lasso_prediction:.2f}")

x = diabetes.data
y = diabetes.target
x_scaled = scaler.transform(x)

ridge_mse = mean_squared_error(y, ridge_model.predict(x_scaled))
lasso_mse = mean_squared_error(y, lasso_model.predict(x_scaled))

ridge_r2 = r2_score(y, ridge_model.predict(x_scaled))
lasso_r2 = r2_score(y, lasso_model.predict(x_scaled))

ridge_model = Ridge(alpha=1.0)
lasso_model = Lasso(alpha=1.0)

n_samples = len(y)
p = len(feature_names)

adjusted_ridge_r2 =  1 - (1 - ridge_r2) * (n_samples - 1) / (n_samples - p - 1)
adjusted_lasso_r2 = 1 - (1 - lasso_r2) * (n_samples - 1) / (n_samples - p - 1)

st.subheader("Model Performance Metrics")

st.write(f" **Ridge Regression:**")
st.write(f" MSE: {ridge_mse:.2f}")
st.write(f"R2 Score: {ridge_r2:.2f}")
st.write(f"Adjusted R2 Score: {adjusted_ridge_r2:.2f}")

st.write(f" **Lasso Regression:**")
st.write(f" MSE: {lasso_mse:.2f}")
st.write(f"R2 Score: {lasso_r2:.2f}")
st.write(f"Adjusted R2 Score: {adjusted_lasso_r2:.2f}")


st.write("Lower values indicate lower diabetes progression risk.")
    
    

