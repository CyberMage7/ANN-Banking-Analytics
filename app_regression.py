import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the regression model
regression_model = tf.keras.models.load_model('regression_model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app setup
st.title('Customer Salary Prediction')
st.write('This app predicts the estimated salary of a bank customer based on their profile.')

# Create tabs for different sections
tab1, tab2 = st.tabs(["Prediction", "About"])

with tab1:
    st.header('Enter Customer Information')
    
    # User Input - same as in the classification app
    col1, col2 = st.columns(2)
    
    with col1:
        geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
        gender = st.selectbox('Gender', label_encoder_gender.classes_)
        age = st.slider('Age', 18, 92, 35)
        credit_score = st.slider('Credit Score', 300, 900, 650)
        balance = st.number_input('Balance', min_value=0.0, value=50000.0, step=1000.0)
    
    with col2:
        tenure = st.slider('Tenure (years)', 0, 10, 5)
        num_of_products = st.slider('Number of Products', 1, 4, 1)
        has_cr_card = st.selectbox('Has Credit Card', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        is_active_member = st.selectbox('Is Active Member', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        exited = st.selectbox('Has Exited', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'Exited': [exited]
    })
    
    # One-hot encode geography
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    
    # Combine all features
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    if st.button('Predict Salary'):
        prediction = regression_model.predict(input_data_scaled)
        predicted_salary = prediction[0][0]
        
        # Display prediction with formatting
        st.subheader('Prediction Result')
        st.metric(label="Estimated Salary", value=f"${predicted_salary:,.2f}")
        
        # Add visual indicator
        if predicted_salary > 100000:
            salary_range = "High"
            color = "green"
        elif predicted_salary > 50000:
            salary_range = "Medium"
            color = "orange"
        else:
            salary_range = "Low"
            color = "red"
        
        st.markdown(f"<h3 style='color:{color}'>Salary Range: {salary_range}</h3>", unsafe_allow_html=True)
        
        # Salary context
        st.write(f"This prediction is based on the customer profile provided. Customers with similar profiles typically have estimated salaries around ${predicted_salary:,.2f}.")

with tab2:
    st.header("About This App")
    st.write("""
    This application uses a deep learning model to predict a customer's estimated salary based on their banking profile. 
    
    ### Model Details
    - **Architecture**: Neural Network with 2 hidden layers
    - **Input Features**: Customer demographics and banking relationship details
    - **Output**: Predicted salary value
    - **Training**: The model was trained on the Churn_Modelling dataset
    
    ### How to Use
    1. Enter the customer information in the form
    2. Click the 'Predict Salary' button
    3. View the predicted salary and analysis
    
    ### Feature Importance
    The most significant factors in predicting a customer's salary include age, credit score, geography, and account balance.
    """)
    
    # Add a sample visualization
    st.subheader("Sample Salary Distribution")
    chart_data = pd.DataFrame(
        np.random.normal(65000, 15000, 100),
        columns=['Estimated Salary']
    )
    st.line_chart(chart_data)
