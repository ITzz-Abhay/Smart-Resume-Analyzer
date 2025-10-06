import streamlit as st
import requests

st.title("Resume Job Category Predictor")

resume_text = st.text_area("Paste your resume here:")

if st.button("Predict"):
    if resume_text.strip() != "":
        response = requests.post(
            "http://127.0.0.1:8000/predict/",
            json={"text": resume_text}
        )
        if response.status_code == 200:
            st.success(f"Predicted Category: {response.json()['category']}")
        else:
            st.error("Error in prediction")
    else:
        st.warning("Please enter some text!")
