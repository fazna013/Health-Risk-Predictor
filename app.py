import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Health Risk Predictor v2", layout="centered")
st.title("🏥 Public Health Risk Predictor (5-Feature Version)")
st.write("Adjust the 5 health parameters below to see your calculated Risk Score live.")

# 1. Load the models securely
@st.cache_resource
def load_models():
    with open('linear_model.pkl', 'rb') as f:
        lr = pickle.load(f)
    with open('knn_model.pkl', 'rb') as f:
        knn = pickle.load(f)
    return lr, knn

try:
    lr_model, knn_model = load_models()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# 2. Setup Sidebar Options and Main Input Layout
model_choice = st.sidebar.radio("Select Prediction Model:", ("Linear Regression", "KNN"))

st.header("📋 User Health Metrics")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (Years)", 18, 85, 45)
    sleep = st.slider("Daily Sleep Hours", 4.0, 10.0, 7.0, 0.5)
    stress = st.slider("Stress Level (1 - 10)", 1.0, 10.0, 5.0, 0.5)

with col2:
    bmi = st.slider("Body Mass Index (BMI)", 15.0, 40.0, 25.0, 0.1)
    pollution = st.slider("Pollution Exposure Level", 10.0, 100.0, 50.0, 1.0)

# 3. Prediction Action
if st.button("🔮 Calculate Risk Score", use_container_width=True):
    # CRITICAL: Match the exact order of features used during model training
    input_data = pd.DataFrame(
        [[sleep, age, stress, bmi, pollution]], 
        columns=['Sleep_Hours', 'Age', 'Stress_Level', 'Body_Mass_Index', 'Pollution_Level']
    )
    
    try:
        if model_choice == "Linear Regression":
            prediction = lr_model.predict(input_data)[0]
        else:
            prediction = knn_model.predict(input_data)[0]
            
        # Bound the prediction realistically between 0% and 100%
        prediction = max(0.0, min(100.0, prediction))
        
        # 4. Display Results
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
        st.write("Ensure your uploaded `.pkl` model files match the layout of this application.")