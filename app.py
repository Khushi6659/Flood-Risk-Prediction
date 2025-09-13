# # flood_app_final.py
# import streamlit as st
# import numpy as np
# import pandas as pd
# import joblib

# # Load trained model artifacts
# artifacts = joblib.load("flood_stack_artifacts.pkl")
# model = artifacts["stack_model"]
# scaler = artifacts["scaler"]
# encoders = artifacts["encoders"]
# selected_features = artifacts["selected_features"]

# st.title("🌊 Flood Risk Prediction App")
# st.write("Predict flood risk based on environmental and demographic data.")

# # -----------------------------
# # Input fields (initialized to 0)
# # -----------------------------
# latitude = st.number_input("Latitude", value=0.0)
# longitude = st.number_input("Longitude", value=0.0)
# rainfall = st.number_input("Rainfall (mm)", value=0.0)
# temperature = st.number_input("Temperature (°C)", value=0.0)
# humidity = st.number_input("Humidity (%)", value=0.0)
# river_discharge = st.number_input("River Discharge (m³/s)", value=0.0)
# water_level = st.number_input("Water Level (m)", value=0.0)
# elevation = st.number_input("Elevation (m)", value=0.0)

# # Categorical inputs
# land_cover = st.selectbox("Land Cover", options=list(encoders["Land Cover"].classes_))
# soil_type = st.selectbox("Soil Type", options=list(encoders["Soil Type"].classes_))
# population_density = st.number_input("Population Density", value=0.0)
# infrastructure = st.selectbox("Infrastructure", ["Poor", "Good"])
# historical_floods = st.number_input("Historical Floods", value=0.0)

# if st.button("Predict Flood Risk"):
#     # Encode categorical inputs
#     land_cover_enc = encoders["Land Cover"].transform([land_cover])[0]
#     soil_type_enc = encoders["Soil Type"].transform([soil_type])[0]
#     infra_enc = 1 if infrastructure == "Good" else 0

#     # Calculate derived features
#     humid_temp = humidity * temperature
#     elevation_waterratio = elevation / (water_level + 1)  # avoid division by zero

#     # Prepare input dataframe
#     input_dict = {
#         "Latitude": latitude,
#         "Longitude": longitude,
#         "Rainfall (mm)": rainfall,
#         "Temperature (°C)": temperature,
#         "Humidity (%)": humidity,
#         "River Discharge (m³/s)": river_discharge,
#         "Water Level (m)": water_level,
#         "Elevation (m)": elevation,
#         "Land Cover": land_cover_enc,
#         "Soil Type": soil_type_enc,
#         "Population Density": population_density,
#         "Infrastructure": infra_enc,
#         "Historical Floods": historical_floods,
#         "Humid_Temp": humid_temp,
#         "Elevation_WaterRatio": elevation_waterratio
#     }

#     input_df = pd.DataFrame([input_dict])
#     # Keep only selected features
#     input_df = input_df[selected_features]

#     # Scale
#     input_scaled = scaler.transform(input_df)

#     # Predict
#     pred_prob = model.predict_proba(input_scaled)[0][1]  # probability of flood
#     pred_label = model.predict(input_scaled)[0]

#     st.subheader("Prediction Results")
#     st.write(f"Flood Probability: {pred_prob*100:.2f}%")
#     if pred_label == 1:
#         st.error("🚨 High Flood Risk Detected!")
#     else:
#         st.success("✅ Low Flood Risk")

#     st.subheader("Input & Derived Features")
#     st.dataframe(input_df.T.rename(columns={0:"Value"}))


# flood_app_visual.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -----------------------------
# Load model and artifacts
# -----------------------------
artifacts = joblib.load("flood_stack_artifacts.pkl")
stack_model = artifacts["stack_model"]
scaler = artifacts["scaler"]
encoders = artifacts["encoders"]
selected_features = artifacts["selected_features"]

# -----------------------------
# Streamlit app title
# -----------------------------
st.set_page_config(page_title="Flood Risk Prediction", layout="wide")
st.title("🌊 Flood Risk Prediction App")
st.write("Predict flood risk based on environmental and demographic data.")

# -----------------------------
# Sidebar for inputs
# -----------------------------
st.sidebar.header("Enter Environmental & Demographic Data")

def user_inputs():
    latitude = st.sidebar.number_input("Latitude", min_value=-90.0, max_value=90.0, value=0.0)
    longitude = st.sidebar.number_input("Longitude", min_value=-180.0, max_value=180.0, value=0.0)
    rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, value=0.0)
    temperature = st.sidebar.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=0.0)
    humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=0.0)
    river_discharge = st.sidebar.number_input("River Discharge (m³/s)", min_value=0.0, value=0.0)
    water_level = st.sidebar.number_input("Water Level (m)", min_value=0.0, value=0.0)
    elevation = st.sidebar.number_input("Elevation (m)", min_value=0.0, value=0.0)
    land_cover = st.sidebar.selectbox("Land Cover", ["Agricultural", "Forest", "Urban", "Barren"])
    soil_type = st.sidebar.selectbox("Soil Type", ["Clay", "Sandy", "Loam"])
    population_density = st.sidebar.number_input("Population Density", min_value=0.0, value=0.0)
    infrastructure = st.sidebar.selectbox("Infrastructure", ["Poor", "Good"])
    historical_floods = st.sidebar.number_input("Historical Floods", min_value=0, value=0)

    return {
        "Latitude": latitude,
        "Longitude": longitude,
        "Rainfall (mm)": rainfall,
        "Temperature (°C)": temperature,
        "Humidity (%)": humidity,
        "River Discharge (m³/s)": river_discharge,
        "Water Level (m)": water_level,
        "Elevation (m)": elevation,
        "Land Cover": land_cover,
        "Soil Type": soil_type,
        "Population Density": population_density,
        "Infrastructure": infrastructure,
        "Historical Floods": historical_floods
    }

inputs = user_inputs()

# -----------------------------
# Transform inputs
# -----------------------------
def transform_inputs(inputs, encoders, scaler, selected_features):
    df_input = pd.DataFrame([inputs])

    # Encode categorical features
    for col in ["Land Cover", "Soil Type", "Infrastructure"]:
        if col in encoders:
            df_input[col] = encoders[col].transform(df_input[col])
        else:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df_input[col] = le.fit_transform(df_input[col])
            encoders[col] = le

    # Feature engineering
    df_input["Humid_Temp"] = df_input["Humidity (%)"] * df_input["Temperature (°C)"]
    df_input["Elevation_WaterRatio"] = df_input["Elevation (m)"] / (df_input["Water Level (m)"] + 1)

    # Keep only selected features
    df_input_selected = df_input[selected_features]

    # Scale numerical features
    df_input_scaled = scaler.transform(df_input_selected)
    return df_input_scaled, df_input_selected

X_scaled_input, df_selected_input = transform_inputs(inputs, encoders, scaler, selected_features)

# -----------------------------
# Prediction button
# -----------------------------
if st.button("Predict Flood Risk"):
    pred_proba = stack_model.predict_proba(X_scaled_input)[0][1]  # Probability of flood
    pred_class = stack_model.predict(X_scaled_input)[0]

    # Display prediction
    st.subheader("Prediction Result")
    st.write(f"**Flood Probability:** {pred_proba*100:.2f}%")
    if pred_class == 1:
        st.error("🚨 High Flood Risk Detected!")
    else:
        st.success("✅ Low Flood Risk")

    # -----------------------------
    # Feature Visualizations
    # -----------------------------
    st.subheader("Feature Visualization")

    # Bar chart
    st.bar_chart(df_selected_input.T)

    # Line chart
    st.line_chart(df_selected_input.T)

    # # Pie chart for flood probability
    # st.subheader("Flood Probability Pie Chart")
    # risk_data = pd.DataFrame({
    #     'Risk': ['High Risk', 'Low Risk'],
    #     'Probability': [pred_proba, 1-pred_proba]
    # })
    # fig, ax = plt.subplots()
    # ax.pie(risk_data['Probability'], labels=risk_data['Risk'], autopct='%1.1f%%', colors=['red','green'])
    # ax.set_title("Flood Risk Probability")
    # st.pyplot(fig)

    # -----------------------------
    # Download report option
    # -----------------------------
    st.subheader("Download Prediction Report")
    df_report = df_selected_input.copy()
    df_report["Flood_Probability"] = pred_proba
    df_report["Flood_Risk"] = "High" if pred_class==1 else "Low"

    csv = df_report.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download CSV Report",
        data=csv,
        file_name="flood_prediction_report.csv",
        mime="text/csv"
    )
