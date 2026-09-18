import json
import urllib.request
import urllib.error

import streamlit as st

API_URL = st.sidebar.text_input("FastAPI URL", "http://127.0.0.1:8001/predict")

st.title("EV Purchase Prediction")
st.caption("Predict whether a customer is likely to buy an EV 🚗.")

with st.form("ev_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        annual_income = st.number_input("Annual Income (USD)", min_value=1.0, value=80000.0, step=1000.0)
        city_type = st.selectbox("City Type", ["Urban", "Rural", "Semi-Urban"])
        daily_commute = st.number_input("Daily Commute (km)", min_value=0.0, value=25.0, step=1.0)
        cars_owned = st.number_input("Number of Cars Owned", min_value=0, value=1)
        current_car = st.selectbox("Current Car Type", ["Truck", "SUV", "Sedan", "Hatchback"])

    with col2:
        home_charging = st.selectbox("Home Charging Possible", ["Yes", "No"])
        charging_home = st.number_input("Charging Stations Near Home", min_value=0, value=2)
        charging_work = st.number_input("Charging Stations Near Work", min_value=0, value=3)
        environmental_concern = st.slider("Environmental Concern Level", 1, 10, 7)
        subsidy = st.selectbox("Subsidy Available", ["Yes", "No"])
        range_anxiety = st.selectbox("Range Anxiety Level", ["Low", "Medium", "High"])

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "Age": int(age),
        "Gender": gender,
        "Annual_Income_USD": float(annual_income),
        "City_Type": city_type,
        "Daily_Commute_km": float(daily_commute),
        "Number_of_Cars_Owned": int(cars_owned),
        "Current_Car_Type": current_car,
        "Charging_Stations_Near_Home": int(charging_home),
        "Charging_Stations_Near_Work": int(charging_work),
        "Home_Charging_Possible": home_charging,
        "Environmental_Concern_Level": int(environmental_concern),
        "Subsidy_Available": subsidy,
        "Range_Anxiety_Level": range_anxiety,
    }

    try:
        req = urllib.request.Request(
            API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as response:
            result = json.loads(response.read().decode("utf-8"))

        prediction = int(result.get("prediction", -1))

        if prediction == 1:
            st.success("Predicted: Yes, the customer is likely to buy an EV.")
        else:
            st.info("Predicted: No, the customer is unlikely to buy an EV.")

        st.json({"request": payload, "response": result})

    except (urllib.error.HTTPError, urllib.error.URLError, ValueError) as exc:
        st.error(f"API request failed: {exc}")
