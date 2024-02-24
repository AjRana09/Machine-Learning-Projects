import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pickle
import streamlit as s
import base64

pre = pickle.load(open(r"C:\Users\Ajay\Studies\Machine Learning\Pickle File\apple_pre.pkl", "rb"))
model = pickle.load(open(r"C:\Users\Ajay\Studies\Machine Learning\Pickle File\apple_model.pkl", "rb"))

@s.cache_data
def get_base64(file):
    with open(file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_base64("D:/Data Science/Machine Learning/Images/apple-transformed.jpeg")  

page_bg_img = f'''
<style>
[data-testid="stAppViewContainer"] {{
background-image: url("data:image/jpeg;base64,{img}");
background-size: cover;
}}
</style>
'''

s.markdown(page_bg_img, unsafe_allow_html=True)


s.title("Apple Quality Prediction Web Applicaiton")
s.subheader("by Ajay Kumar Maharana")

size = s.number_input("Type Size:")
weight = s.number_input("Type Weight:")
sweetness = s.number_input("Type Sweetness:")
crunchiness = s.number_input("Type Crunchiness:")
juiciness = s.number_input("Type Juciness:")
ripeness  = s.number_input("Type Ripeness:")
acidity = s.number_input("Type Acidity:")

query = pd.DataFrame([[size, weight, sweetness, crunchiness, juiciness, ripeness, acidity]],
                 columns=['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity'])
query_trans = pre.transform(query)
predicted_yi = model.predict(query_trans)

if predicted_yi==1:
    x = "GOOD."
else:
    x = "BAD."

if s.button("Submit"):
    s.write('Your Apple Quality is ' + x )
