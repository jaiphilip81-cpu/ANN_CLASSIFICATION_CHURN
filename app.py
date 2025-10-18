import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# Load the trained model
model = tf.keras.models.load_model('churn_model.h5')

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load the One-Hot Encoder
with open('onehot_encoder.pkl', 'rb') as file:
    ohe = pickle.load(file)

# load the label encoder
with open('label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)


### Streamlit App
st.title("Customer Churn Prediction")

geography = st.selectbox("Geography", ohe.categories_[0])
gender = st.selectbox("Gender", le.classes_)
credit_score = st.slider("Credit Score", 300, 850, 600)
age = st.slider("Age", 18, 100, 30)
tenure = st.slider("Tenure (years)", 0, 10, 3)
balance = st.number_input("Balance", min_value=0.0, value=1000.0)
num_of_products = st.slider("Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [le.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]    
})

# Preprocess the input data
geography_encoded = ohe.transform(input_data[['Geography']]).toarray()

geography_encoded_df = pd.DataFrame(geography_encoded, columns=ohe.get_feature_names_out(['Geography']))
input_df = pd.concat([ input_data.drop('Geography', axis=1), geography_encoded_df], axis=1)

input_scaled = scaler.transform(input_df)

# Make prediction
prediction = model.predict(input_scaled)
churn_probability = prediction[0][0]
st.subheader("Churn Probability")
st.write(f"{churn_probability:.2%}")
# Display result
if churn_probability > 0.5:
    st.error("The customer is likely to churn.")
else:
    st.success("The customer is unlikely to churn.")
