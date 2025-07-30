import joblib
import streamlit as st
import pandas as pd

# Load the model
@st.cache_resource
def load_model():
    model = joblib.load("titanic_model.pkl")
    return model

model = load_model()

# App title and description
st.title("🚢 Titanic Survival Prediction App")
st.markdown("Enter passenger details to predict survival.")

# Input fields
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.radio("Sex", ("Male", "Female"))
age = st.slider("Age", min_value=1, max_value=80, value=25)
sibsp = st.selectbox("# of Siblings/Spouses Aboard", range(0, 6))
parch = st.selectbox("# of Parents/Children Aboard", range(0, 6))
fare = st.number_input("Ticket Fare", min_value=0.0, max_value=600.0, value=32.0)
embarked = st.selectbox("Port of Embarkation", ("S", "C", "Q"))

# Encode inputs
sex_encoded = 0 if sex == "Male" else 1
embarked_encoded = {"S": 0, "C": 1, "Q": 2}[embarked]

# Create DataFrame for prediction
input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex_encoded],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked_encoded]
})

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][prediction]

    if prediction == 1:
        st.success(f"The passenger is likely to **Survive** (Confidence: {prediction_proba:.2%})")
    else:
        st.error(f"The passenger is likely to **Not Survive** (Confidence: {prediction_proba:.2%})")
