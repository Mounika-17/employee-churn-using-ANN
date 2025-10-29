import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# Streamlit app title
st.title('Employee Churn Prediction App')

st.markdown(
    """
    This app predicts whether a customer is likely to churn based on their details.
    Please provide the information below:
    """
)

# Collect user input
credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=600)
geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', min_value=18, max_value=92, value=40)
tenure = st.slider('Tenure (years)', min_value=0, max_value=10, value=3)
balance = st.number_input('Account Balance', min_value=0.0, step=1000.0, value=60000.0)
num_of_products = st.slider('Number of Products', min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox('Has Credit Card?', [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
is_active_member = st.selectbox('Is Active Member?', [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, step=1000.0, value=50000.0)

# When user clicks "Predict"
if st.button('Predict Churn'):
    try:
        # Create a CustomData object
        custom_data = CustomData(
            CreditScore=credit_score,
            Geography=geography,
            Gender=gender,
            Age=age,
            Tenure=tenure,
            Balance=balance,
            NumOfProducts=num_of_products,
            HasCrCard=has_cr_card,
            IsActiveMember=is_active_member,
            EstimatedSalary=estimated_salary
        )

        # Convert input to DataFrame
        input_df = custom_data.get_data_as_data_frame()

        # Initialize and run prediction pipeline
        predict_pipeline = PredictPipeline()
        churn_label, churn_prob = predict_pipeline.predict(input_df)

        # Display results
        st.subheader("📊 Prediction Results")
        st.write(f"**Churn Probability:** {churn_prob:.2f}")

        if churn_label == 1:
            st.error("🚨 The customer is **likely to churn.**")
        else:
            st.success("✅ The customer is **not likely to churn.**")

    except Exception as e:
        st.error(f"Error: {e}")
