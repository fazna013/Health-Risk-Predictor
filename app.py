import streamlit as st
import pandas as pd
import pickle

# Set page configuration
st.set_page_config(page_title="Health Risk Predictor v2", layout="centered")

# --- SESSION STATE INITIALIZATION ---
if "page" not in st.session_state:
    st.session_state.page = "welcome"

# --- PAGE 1: WELCOME PAGE ---
if st.session_state.page == "welcome":
    st.title("🏥 Public Health Risk Predictor")
    st.write("---")
    
    st.write("Click the button below to access the prediction dashboard.")
    
    if st.button("🚀 Go to Dashboard", use_container_width=True):
        st.session_state.page = "dashboard"
        st.rerun()

# --- PAGE 2: PREDICTION DASHBOARD ---
elif st.session_state.page == "dashboard":
    
    if st.sidebar.button("⬅️ Back to Welcome Page"):
        st.session_state.page = "welcome"
        st.rerun()

    st.title("📊 Health Metrics Dashboard")
    st.write("Adjust the 5 parameters below to see your calculated Risk Score live (Powered by Linear Regression).")

    # 1. Load ONLY the Linear Regression model securely
    @st.cache_resource
    def load_linear_model():
        with open('linear_model.pkl', 'rb') as f:
            lr = pickle.load(f)
        return lr

    try:
        lr_model = load_linear_model()
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        st.stop()

    st.header("📋 User Health Metrics")
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age (Years)", 18, 85, 45)
        sleep = st.slider("Daily Sleep Hours", 4.0, 10.0, 7.0, 0.5)
        stress = st.slider("Stress Level (1 - 10)", 1.0, 10.0, 5.0, 0.5)

    with col2:
        bmi = st.slider("Body Mass Index (BMI)", 15.0, 40.0, 25.0, 0.1)
        pollution = st.slider("Pollution Exposure Level", 10.0, 100.0, 50.0, 1.0)

    # 2. Prediction Action
    if st.button("🔮 Calculate Risk Score", use_container_width=True):
        # Match the exact column order used during model training
        input_data = pd.DataFrame(
            [[sleep, age, stress, bmi, pollution]], 
            columns=['Sleep_Hours', 'Age', 'Stress_Level', 'Body_Mass_Index', 'Pollution_Level']
        )
        
        try:
            # Generate prediction using only the Linear Regression model
            prediction = lr_model.predict(input_data)[0]
                
            # Bound the prediction realistically between 0% and 100%
            prediction = max(0.0, min(100.0, prediction))
            
            # 3. Display Results
            st.markdown("---")
            st.subheader(f"Predicted Risk Score: {prediction:.1f}%")
            
            if prediction < 45:
                st.success("✅ Low Risk Profile")
            elif 45 <= prediction < 70:
                st.warning("⚠️ Moderate Risk Profile")
            else:
                st.error("🚨 High Risk Profile")
                
        except ValueError as e:
            st.error("🚨 Feature Mismatch Error Detected!")
            st.write("Ensure your uploaded `linear_model.pkl` file matches the layout of this application.")