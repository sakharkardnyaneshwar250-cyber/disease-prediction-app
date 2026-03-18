import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --------- LOAD DATA ---------
data = pd.read_csv("diabetes.csv")

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# --------- TRAIN MODEL ---------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# --------- UI ---------
st.set_page_config(page_title="Disease Prediction App", page_icon="🏥")

st.title("Disease Prediction App 🏥")
st.subheader("Enter patient details below 👇")

# --------- INPUTS (SLIDERS) ---------
preg = st.slider("Pregnancies", 1, 5, 2)
glucose = st.slider("Glucose", 80, 180, 100)
bp = st.slider("Blood Pressure", 60, 120, 80)
skin = st.slider("Skin Thickness", 10, 50, 20)
insulin = st.slider("Insulin", 50, 300, 100)
bmi = st.slider("BMI", 18, 40, 25)
dpf = st.slider("Diabetes Pedigree Function", 0.1, 2.5, 1.0)
age = st.slider("Age", 20, 70, 30)

# --------- PREDICTION ---------
if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")