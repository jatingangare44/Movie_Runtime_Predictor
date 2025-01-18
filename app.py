import streamlit as st
import pickle
import numpy as np
import sklearn

# Load the model and scaler
model = pickle.load(open('runtime_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Streamlit app header
st.title("Movie Runtime Prediction")

# Input fields
budget = st.number_input("Enter budget (in million):", value=180.0, min_value=0.0)
vote_average = st.number_input("Enter vote average:", value=7.8, min_value=0.0)

# Predict button
if st.button("Predict Runtime"):
    try:
        # Multiply budget by 1,000,000 (as required in your code)
        budget = budget * 1000000

        # Prepare the input data
        new_input = np.array([[budget, vote_average]])
        new_input_scaled = scaler.transform(new_input)

        # Predict the runtime using the model
        predicted_runtime = model.predict(new_input_scaled)

        # Show the result
        st.write(f"Predicted runtime: {predicted_runtime[0]:.2f} minutes")

    except Exception as e:
        st.error(f"Error: {e}")
