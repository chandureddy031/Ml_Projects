import streamlit as st
import pandas as pd
import joblib

model = joblib.load("linear_reg_pipeline.joblib")

st.set_page_config(page_title="ML Prediction App", layout="centered")

st.title("Linear Regression Prediction")
st.write("Enter feature values to get prediction")

col1 = st.number_input("Feature 1", value=24.50)
col2 = st.number_input("Feature 2", value=1)
col3 = st.number_input("Feature 3", value=0)
col4 = st.number_input("Feature 4", value=0)
col5 = st.number_input("Feature 5", value=1)
col6 = st.number_input("Feature 6", value=4)

if st.button("Predict"):
    input_df = pd.DataFrame(
        [[col1, col2, col3, col4, col5, col6]],
        columns=model.named_steps["preprocess"].feature_names_in_
    )

    prediction = model.predict(input_df)

    st.success(f"Prediction: {prediction[0]}")
