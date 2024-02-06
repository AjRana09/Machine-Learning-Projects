# Importing libraries
import numpy as np
import pandas as pd
import sklearn
import streamlit as s
import pickle

# Load pre-processing and model files
pre = pickle.load(open(r"C:\Users\Ajay\Studies\Machine Learning\Assignment\drugs_pre.pkl", "rb"))
model = pickle.load(open(r"C:\Users\Ajay\Studies\Machine Learning\Assignment\drugs_model.pkl", "rb"))

# Streamlit title 
s.title("Drug Classification")

# Age input
age = s.slider('**Enter Your Age (yrs.) :** ')

# Gender input
sex = s.radio(
    "**Gender**",
    ["M", "F"],
    horizontal=True
)

# Blood Pressure input
bp = s.radio(
    "**Blood Pressure**",
    ["HIGH", "NORMAL", "LOW"],
    horizontal=True
)

# Cholesterol input
cholesterol = s.radio(
    "**Cholesterol**",
    ["HIGH", "NORMAL"],
    horizontal=True
)

# Na_to_K input
na_to_k = s.slider('**Enter Your Na_to_K :** ', min_value= 0.0, max_value= 100.0, step= 0.01)

# Create DataFrame for prediction
query = pd.DataFrame([[age, sex, bp, cholesterol, na_to_k]],
                 columns=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'])

# Transform input data
query_trans = pre.transform(query)

# Predict and display result
predicted_yi = model.predict(query_trans)

if s.button('Submit'):
    if predicted_yi == 0:
        result = "**DrugY**"
    elif predicted_yi == 1:
        result = "**drugA**"
    elif predicted_yi == 2:
        result = "**drugB**"
    elif predicted_yi == 3:
        result = "**drugC**"
    elif predicted_yi == 4:
        result = "**drugX**"
    else:
        result = "**Error**"

    s.write(f"Based on the input, the predicted drug is: {result}")