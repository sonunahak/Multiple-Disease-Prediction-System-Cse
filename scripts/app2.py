import pickle
import streamlit as st
from streamlit_option_menu import option_menu

#loading models  
with open(r"C:\Users\khuza\OneDrive\Desktop\Data science course\PROJECTS\MULTIPLE DISEASE PREDICTION\models\diabetes_model.sav", 'rb') as f:
    diabetes_model = pickle.load(f)

with open(r"C:\Users\khuza\OneDrive\Desktop\Data science course\PROJECTS\MULTIPLE DISEASE PREDICTION\models\heart_disease_model.sav", 'rb') as f:
    heart_disease_model = pickle.load(f)

with open(r"C:\Users\khuza\OneDrive\Desktop\Data science course\PROJECTS\MULTIPLE DISEASE PREDICTION\models\parkinsons_model.sav", 'rb') as f:
    parkinsons_model = pickle.load(f)

with open(r"C:\Users\khuza\OneDrive\Desktop\Data science course\PROJECTS\MULTIPLE DISEASE PREDICTION\models\breast_cancer_model.pkl", 'rb') as f:
    breast_cancer_model = pickle.load(f)


# Sidebar menu
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 'Breast Cancer Prediction'],
        icons=['activity', 'heart', 'person', 'gender-female'],
        menu_icon='hospital',
        default_index=0
    )

# Styling
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    </style>
    """,
    unsafe_allow_html=True
)

# Pages
if selected == 'Diabetes Prediction':
    st.title("🩸 Diabetes Prediction")
    st.write("Enter medical details to check the likelihood of having diabetes.")

    with st.form("diabetes_form"):
        col1, col2, col3 = st.columns(3)
        with col1: Pregnancies = st.number_input('Pregnancies', min_value=0, step=1, value=None, help="Type here")
        with col2: Glucose = st.number_input('Glucose Level', min_value=0, step=1, value=None, help="Type here")
        with col3: BloodPressure = st.number_input('Blood Pressure', min_value=0, step=1, value=None, help="Type here")

        col4, col5, col6 = st.columns(3)
        with col4: SkinThickness = st.number_input('Skin Thickness', min_value=0, step=1, value=None, help="Type here")
        with col5: Insulin = st.number_input('Insulin Level', min_value=0, step=1, value=None, help="Type here")
        with col6: BMI = st.number_input('BMI', min_value=0.0, step=0.1, value=None, help="Type here")

        col7, col8 = st.columns(2)
        with col7: DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, step=0.01, value=None, help="Type here")
        with col8: Age = st.number_input('Age', min_value=0, step=1, value=None, help="Type here")

        submit_btn = st.form_submit_button("Predict")

    if submit_btn:
        if None in [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]:
            st.warning("⚠ Please fill in all fields.")
        else:
            diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            st.success("✅ The person is Diabetic" if diab_prediction[0] == 1 else "✅ The person is NOT Diabetic")

elif selected == 'Heart Disease Prediction':
    st.title("❤️ Heart Disease Prediction")
    st.write("Provide your medical details to assess heart disease risk.")

    with st.form("heart_form"):
        col1, col2, col3 = st.columns(3)
        with col1: age = st.number_input('Age', min_value=0, step=1, value=None, help="Type here")
        with col2: sex = st.number_input('Sex (1=Male, 0=Female)', min_value=0, max_value=1, step=1, value=None, help="Type here")
        with col3: cp = st.number_input('Chest Pain Type (0-3)', min_value=0, max_value=3, step=1, value=None, placeholder="Type here")

        col4, col5, col6 = st.columns(3)
        with col4: trestbps = st.number_input('Resting Blood Pressure', min_value=0, step=1, value=None, help="Type here")
        with col5: chol = st.number_input('Serum Cholestoral', min_value=0, step=1, value=None, help="Type here")
        with col6: fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)', min_value=0, max_value=1, step=1, value=None, help="Type here")

        col7, col8, col9 = st.columns(3)
        with col7: restecg = st.number_input('Resting Electrocardiographic (0-2)', min_value=0, max_value=2, step=1, value=None, help="Type here")
        with col8: thalach = st.number_input('Max Heart Rate Achieved', min_value=0, step=1, value=None, help="Type here")
        with col9: exang = st.number_input('Exercise Induced Angina (1=True, 0=False)', min_value=0, max_value=1, step=1, value=None, help="Type here")

        col10, col11, col12 = st.columns(3)
        with col10: oldpeak = st.number_input('ST Depression', step=0.1, value=None, help="Type here")
        with col11: slope = st.number_input('Slope of ST segment (0-2)', min_value=0, max_value=2, step=1, value=None, help="Type here")
        with col12: ca = st.number_input('Major vessels colored (0-4)', min_value=0, max_value=4, step=1, value=None, help="Type here")

        thal = st.number_input('Thalassemia (1=Normal, 2=Fixed Defect, 3=Reversible Defect)', min_value=1, max_value=3, step=1, value=None, help="Type here")

        submit_btn = st.form_submit_button("Predict")

    if submit_btn:
        if None in [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]:
            st.warning("⚠ Please fill in all fields.")
        else:
            heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            st.success("✅ The person has Heart Disease" if heart_prediction[0] == 1 else "✅ The person does NOT have Heart Disease")

elif selected == "Parkinsons Prediction":
    st.title("🧠 Parkinson's Disease Prediction")
    st.write("Fill in the speech and biomedical measurements to detect Parkinson's disease.")

    with st.form("parkinsons_form"):
        parkinson_inputs = []
        cols = st.columns(5)
        for i in range(22):
            with cols[i % 5]:
                parkinson_inputs.append(st.number_input(f"Feature {i+1}", value=None, placeholder="Type here"))

        submit_btn = st.form_submit_button("Predict")

    if submit_btn:
        if None in parkinson_inputs:
            st.warning("⚠ Please fill in all fields.")
        else:
            parkinsons_prediction = parkinsons_model.predict([parkinson_inputs])
            st.success("✅ The person has Parkinson's Disease" if parkinsons_prediction[0] == 1 else "✅ The person does NOT have Parkinson's Disease")

elif selected == "Breast Cancer Prediction":
    st.title("🎀 Breast Cancer Prediction")
    st.write("Provide tumor measurement details to check for malignancy.")

    with st.form("cancer_form"):
        cancer_inputs = []
        cols = st.columns(5)
        for i in range(30):
            with cols[i % 5]:
                cancer_inputs.append(st.number_input(f"Feature {i+1}", value=None, placeholder="Type here"))

        submit_btn = st.form_submit_button("Predict")

    if submit_btn:
        if None in cancer_inputs:
            st.warning("⚠ Please fill in all fields.")
        else:
            cancer_prediction = breast_cancer_model.predict([cancer_inputs])
            st.success("✅ The tumor is Malignant" if cancer_prediction[0] == 1 else "✅ The tumor is Benign")

st.markdown(
    """
    <hr style="margin-top:50px;">
    <div style="text-align:center; color:gray;">
        Developed by <b>Huzaif Ulla Khan</b> 🧑‍💻
    </div>
    """,
    unsafe_allow_html=True
)