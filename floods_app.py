import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import init_db, add_user, verify_user, save_input

# Initialize DB
init_db()

# Session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# Sample flood prediction (mock)
def predict_flood(input_data):
    score = (
        input_data["Rainfall"]*0.3 +
        input_data["River Discharge"]*0.25 +
        input_data["Water Level"]*0.2 +
        input_data["Historical Floods"]*0.25
    )
    probability = min(score / 100, 1.0)
    if probability < 0.3:
        risk = "Low Flood Risk"
    elif probability < 0.7:
        risk = "Medium Flood Risk"
    else:
        risk = "High Flood Risk"
    return probability*100, risk

# --- Registration Page ---
def registration():
    st.title("Register")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Register"):
        if username == "" or password == "":
            st.error("Please fill both fields.")
        elif add_user(username, password):
            st.success("Registered successfully! Please login.")
        else:
            st.error("Username already exists. Choose another.")

# --- Login Page ---
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if verify_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome {username}!")
        else:
            st.error("Invalid username or password.")

# --- Prediction Page ---
def prediction():
    st.title("🌊 Flood Risk Prediction")
    st.write(f"Logged in as: {st.session_state.username}")
    
    input_data = {}
    input_data["Latitude"] = st.number_input("Latitude", 0.0)
    input_data["Longitude"] = st.number_input("Longitude", 0.0)
    input_data["Rainfall"] = st.number_input("Rainfall (mm)", 0.0)
    input_data["Temperature"] = st.number_input("Temperature (°C)", 0.0)
    input_data["Humidity"] = st.number_input("Humidity (%)", 0.0)
    input_data["River Discharge"] = st.number_input("River Discharge (m³/s)", 0.0)
    input_data["Water Level"] = st.number_input("Water Level (m)", 0.0)
    input_data["Elevation"] = st.number_input("Elevation (m)", 0.0)
    input_data["Land Cover"] = st.selectbox("Land Cover", ["Agricultural", "Urban", "Forest"])
    input_data["Soil Type"] = st.selectbox("Soil Type", ["Clay", "Sandy", "Loamy"])
    input_data["Population Density"] = st.number_input("Population Density", 0.0)
    input_data["Infrastructure"] = st.selectbox("Infrastructure", ["Poor", "Average", "Good"])
    input_data["Historical Floods"] = st.number_input("Historical Floods", 0.0)
    
    if st.button("Predict Flood Risk"):
        save_input(st.session_state.username, input_data)
        probability, risk = predict_flood(input_data)
        st.subheader("Prediction Result")
        st.write(f"Flood Probability: {probability:.2f}%")
        if risk == "Low Flood Risk":
            st.success(risk)
        elif risk == "Medium Flood Risk":
            st.warning(risk)
        else:
            st.error(risk)
        
        # Visualization
        st.subheader("Feature Contribution")
        labels = list(input_data.keys())[:4]
        values = [input_data[k] for k in labels]
        fig, ax = plt.subplots()
        ax.bar(labels, values, color='skyblue')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Download report
        st.subheader("Download Report")
        df_report = pd.DataFrame([input_data])
        df_report["Flood Probability"] = probability
        df_report["Risk Level"] = risk
        csv = df_report.to_csv(index=False).encode()
        st.download_button("Download CSV", data=csv, file_name="flood_report.csv", mime="text/csv")

# --- Main ---
if not st.session_state.logged_in:
    tab = st.sidebar.selectbox("Navigation", ["Login", "Register"])
    if tab == "Login":
        login()
    else:
        registration()
else:
    prediction()
