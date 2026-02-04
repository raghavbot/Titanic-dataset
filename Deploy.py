import streamlit as st
import joblib
import numpy as np

st.title("Titanic Survival Prediction")

model = joblib.load("titanic_model.joblib")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=100, value=30)
SibSp = st.number_input("Number of Siblings/Spouses on board", min_value=0, max_value=10, value=0)
Parch = st.number_input("Number of Parents/Children on board", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=50.0)
Embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

sex_val = 0 if sex == "Male" else 1
embarked_val = {"C": 1, "Q": 2, "S": 0}[Embarked]

sample = np.array([[pclass, sex_val, age, SibSp, Parch, fare, embarked_val]])

if st.button("Predict"):
    prediction = model.predict(sample)[0]

    if prediction == 1:
        st.success("Passenger would SURVIVE")
    else:
        st.error("Passenger would NOT survive")
