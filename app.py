import streamlit as st
import tensorflow as tf
# from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np

model=tf.keras.models.load_model("model.h5")

with open("scaler.pkl","rb") as file:
    scaler=pickle.load(file)
    
with open("onehot_encoder_geo.pkl","rb") as file:
    onehot_encoder_geo=pickle.load(file)
    
with open("label_encoder_gender.pkl","rb") as file:
    label_encoder_gender=pickle.load(file)
    
#Streamlit App
st.title("Customer Churn Prediction")
    
st.title('Customer Churn Prediction')
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 100, 30)
balance = st.number_input('Balance', min_value=0.0, value=0.0)
credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=600)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)
tenure = st.slider('Tenure', 0, 10, 5)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


geo_encoded = onehot_encoder_geo.transform([[input_data[geography]]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out([geography]))

input_data=pd.concat([input_data.drop("Geography",axis=1),geo_encoded_df],axis=1)
#Scale churn 
input_scaled=scaler.transform(input_data)

#Predict churn
prediction=model.predict(input_scaled)
prediction_probability=prediction[0][0]

if st.button("Predict Churn"):
    if prediction_probability>0.5:
        st.error(f"The customer is likely to churn with a probability of {prediction_probability:.2f}")
    else:
        st.success(f"The customer is unlikely to churn with a probability of {1-prediction_probability:.2f}")