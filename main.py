import pickle
import numpy as np
import streamlit as st

# Load the model
with open("best_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
vectorizer = data["vectorizer"]
label_encoder = data["label_encoder"]

st.title("Disease Prediction from Symptoms")

# User input
symptoms = st.text_area("Enter symptoms (comma-separated):")

if st.button("Predict"):
    if symptoms:
        # Transform input symptoms
        symptoms_transformed = vectorizer.transform([symptoms])

        # Predict disease
        predicted_disease = label_encoder.inverse_transform(model.predict(symptoms_transformed))[0]
        
        # Get top 3 probable diseases
        probabilities = model.decision_function(symptoms_transformed)
        top_3_indices = np.argsort(probabilities, axis=1)[:, -3:]
        top_3_diseases = label_encoder.inverse_transform(top_3_indices[0])

        st.success(f"Most Likely Disease: **{predicted_disease}**")
        st.write(f"Top 3 Possible Diseases: {', '.join(top_3_diseases)}")
    else:
        st.warning("Please enter symptoms.")

