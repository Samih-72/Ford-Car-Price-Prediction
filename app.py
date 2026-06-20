import streamlit as st
import pandas as pd
import joblib

# Load dataset
df = pd.read_csv("ford.csv")

# Load trained model
model = joblib.load("model.pkl")

st.title("🚗 Ford Car Price Predictor")

# Show dataset
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Visualization
st.subheader("Mileage Distribution")
st.bar_chart(df["mileage"].head(50))

st.subheader("Price Distribution")
st.line_chart(df["price"])

# Prediction Section
st.subheader("Predict Car Price")

year = st.number_input(
    "Year",
    min_value=2000,
    max_value=2025,
    value=2018
)

mileage = st.number_input(
    "Mileage",
    min_value=0,
    value=50000
)

engine_size = st.number_input(
    "Engine Size",
    min_value=1.0,
    max_value=5.0,
    value=2.0
)

if st.button("Predict Price"):

    prediction = model.predict(
        [[year, mileage, engine_size]]
    )

    st.success(
        f"Estimated Price: £{prediction[0]:,.2f}"
    )