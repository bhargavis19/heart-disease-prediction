import streamlit as st
import pandas as pd
import joblib

# Define the prepare_data function
def prepare_data(sysBP, glucose, age, totChol, cigsPerDay, diaBP, prevalentHyp, diabetes, BPMeds, male):
    data = {
        'sysBP': [sysBP],
        'glucose': [glucose],
        'age': [age],
        'totChol': [totChol],
        'cigsPerDay': [cigsPerDay],
        'diaBP': [diaBP],
        'prevalentHyp': [prevalentHyp],
        'diabetes': [diabetes],
        'BPMeds': [BPMeds],
        'male': [male]
    }
    X = pd.DataFrame(data)
    return X

# Load the joblib file containing the tuned model
joblib_file_path = 'tuned_adaboost_model.joblib'
loaded_model = joblib.load(joblib_file_path)

# Define a function to make predictions using the loaded model
def predict_heart_disease(X):
    return loaded_model.predict(X)

# Main function to run the Streamlit app
def main():
    st.title("Heart Disease Prediction App")

    # Input fields for variables
    sysBP = st.number_input("Enter sysBP")
    glucose = st.number_input("Enter glucose")
    age = st.number_input("Enter age")
    totChol = st.number_input("Enter totChol")
    cigsPerDay = st.number_input("Enter cigsPerDay")
    diaBP = st.number_input("Enter diaBP")
    prevalentHyp = st.selectbox("prevalentHyp", [0, 1])
    diabetes = st.selectbox("diabetes", [0, 1])
    BPMeds = st.selectbox("BPMeds", [0, 1])
    male = st.selectbox("male", [0, 1])

    # Button to trigger data preparation and prediction
    if st.button("Predict Heart Disease"):
        # Call prepare_data function
        X = prepare_data(sysBP, glucose, age, totChol, cigsPerDay, diaBP, prevalentHyp, diabetes, BPMeds, male)
        st.write("Input Data:")
        st.write(X)

        # Make prediction using the loaded model
        prediction = predict_heart_disease(X)
        st.write("Prediction:")
        st.write(prediction)

# Run the app
if __name__ == "__main__":
    main()