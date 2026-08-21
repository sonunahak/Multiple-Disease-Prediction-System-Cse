import datetime
import os
import pickle
import sqlite3
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu


# ==========================================
# Database Initialization & Helpers
# ==========================================
def init_db():
  conn = sqlite3.connect('patient_records.db')
  c = conn.cursor()
  c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            disease_type TEXT,
            prediction_result TEXT
        )
    """)
  conn.commit()
  conn.close()


def save_prediction(disease_type, result):
  conn = sqlite3.connect('patient_records.db')
  c = conn.cursor()
  timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  c.execute(
      """
        INSERT INTO history (timestamp, disease_type, prediction_result)
        VALUES (?, ?, ?)
    """,
      (timestamp, disease_type, result),
  )
  conn.commit()
  conn.close()


# Initialize SQLite Database
init_db()


# ==========================================
# Streamlit App Configuration
# ==========================================
st.set_page_config(
    page_title='Health Assistant', layout='wide', page_icon='🧑‍⚕️'
)

# Model Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

# Load Machine Learning Models
with open(os.path.join(MODELS_DIR, 'diabetes_model.sav'), 'rb') as f:
  diabetes_model = pickle.load(f)

with open(os.path.join(MODELS_DIR, 'heart_disease_model.sav'), 'rb') as f:
  heart_disease_model = pickle.load(f)

with open(os.path.join(MODELS_DIR, 'parkinsons_model.sav'), 'rb') as f:
  parkinsons_model = pickle.load(f)

with open(os.path.join(MODELS_DIR, 'breast_cancer_model.pkl'), 'rb') as f:
  breast_cancer_model = pickle.load(f)


# ==========================================
# Sidebar Navigation Menu
# ==========================================
with st.sidebar:
  selected = option_menu(
      'Multiple Disease Prediction System',
      [
          'Diabetes Prediction',
          'Heart Disease Prediction',
          'Parkinsons Prediction',
          'Breast Cancer Prediction',
          'Patient History',
      ],
      icons=['activity', 'heart', 'person', 'droplet-half', 'clock-history'],
      menu_icon='hospital-fill',
      default_index=0,
  )


# ==========================================
# 1. Diabetes Prediction Page
# ==========================================
if selected == 'Diabetes Prediction':
  st.title('Diabetes Prediction using ML')

  col1, col2, col3 = st.columns(3)

  with col1:
    Pregnancies = st.text_input('Number of Pregnancies', '0')
  with col2:
    Glucose = st.text_input('Glucose Level', '120')
  with col3:
    BloodPressure = st.text_input('Blood Pressure value', '70')

  with col1:
    SkinThickness = st.text_input('Skin Thickness value', '20')
  with col2:
    Insulin = st.text_input('Insulin Level', '80')
  with col3:
    BMI = st.text_input('BMI value', '25.0')

  with col1:
    DiabetesPedigreeFunction = st.text_input(
        'Diabetes Pedigree Function value', '0.5'
    )
  with col2:
    Age = st.text_input('Age of the Person', '30')

  if st.button('Diabetes Test Result'):
    try:
      user_input = [
          float(Pregnancies),
          float(Glucose),
          float(BloodPressure),
          float(SkinThickness),
          float(Insulin),
          float(BMI),
          float(DiabetesPedigreeFunction),
          float(Age),
      ]
      diab_prediction = diabetes_model.predict([user_input])

      if diab_prediction[0] == 1:
        diab_diagnosis = 'The person is diabetic'
        st.error(diab_diagnosis)
      else:
        diab_diagnosis = 'The person is not diabetic'
        st.success(diab_diagnosis)

      save_prediction('Diabetes', diab_diagnosis)
    except Exception as e:
      st.warning('Please enter valid numeric values in all fields.')


# ==========================================
# 2. Heart Disease Prediction Page
# ==========================================
if selected == 'Heart Disease Prediction':
  st.title('Heart Disease Prediction using ML')

  col1, col2, col3 = st.columns(3)

  with col1:
    age = st.text_input('Age', '50')
  with col2:
    sex = st.text_input('Sex (1 = Male, 0 = Female)', '1')
  with col3:
    cp = st.text_input('Chest Pain types (0, 1, 2, 3)', '0')

  with col1:
    trestbps = st.text_input('Resting Blood Pressure', '120')
  with col2:
    chol = st.text_input('Serum Cholestoral in mg/dl', '200')
  with col3:
    fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)', '0')

  with col1:
    restecg = st.text_input('Resting Electrocardiographic results (0, 1, 2)', '0')
  with col2:
    thalach = st.text_input('Maximum Heart Rate achieved', '150')
  with col3:
    exang = st.text_input('Exercise Induced Angina (1 = Yes, 0 = No)', '0')

  with col1:
    oldpeak = st.text_input('ST depression induced by exercise', '1.0')
  with col2:
    slope = st.text_input('Slope of the peak exercise ST segment', '1')
  with col3:
    ca = st.text_input('Major vessels colored by flourosopy (0-3)', '0')

  with col1:
    thal = st.text_input(
        'thal: 0 = normal; 1 = fixed defect; 2 = reversable defect', '1'
    )

  if st.button('Heart Disease Test Result'):
    try:
      user_input = [
          float(age),
          float(sex),
          float(cp),
          float(trestbps),
          float(chol),
          float(fbs),
          float(restecg),
          float(thalach),
          float(exang),
          float(oldpeak),
          float(slope),
          float(ca),
          float(thal),
      ]
      heart_prediction = heart_disease_model.predict([user_input])

      if heart_prediction[0] == 1:
        heart_diagnosis = 'The person is having heart disease'
        st.error(heart_diagnosis)
      else:
        heart_diagnosis = 'The person does not have any heart disease'
        st.success(heart_diagnosis)

      save_prediction('Heart Disease', heart_diagnosis)
    except Exception as e:
      st.warning('Please enter valid numeric values in all fields.')


# ==========================================
# 3. Parkinson's Prediction Page
# ==========================================
if selected == 'Parkinsons Prediction':
  st.title("Parkinson's Disease Prediction using ML")

  col1, col2, col3, col4, col5 = st.columns(5)

  with col1:
    fo = st.text_input('MDVP:Fo(Hz)', '119.99')
  with col2:
    fhi = st.text_input('MDVP:Fhi(Hz)', '157.30')
  with col3:
    flo = st.text_input('MDVP:Flo(Hz)', '74.99')
  with col4:
    Jitter_percent = st.text_input('MDVP:Jitter(%)', '0.0078')
  with col5:
    Jitter_Abs = st.text_input('MDVP:Jitter(Abs)', '0.00007')

  with col1:
    RAP = st.text_input('MDVP:RAP', '0.0037')
  with col2:
    PPQ = st.text_input('MDVP:PPQ', '0.0055')
  with col3:
    DDP = st.text_input('Jitter:DDP', '0.011')
  with col4:
    Shimmer = st.text_input('MDVP:Shimmer', '0.0437')
  with col5:
    Shimmer_dB = st.text_input('MDVP:Shimmer(dB)', '0.426')

  with col1:
    APQ3 = st.text_input('Shimmer:APQ3', '0.0218')
  with col2:
    APQ5 = st.text_input('Shimmer:APQ5', '0.0313')
  with col3:
    APQ = st.text_input('MDVP:APQ', '0.0297')
  with col4:
    DDA = st.text_input('Shimmer:DDA', '0.0654')
  with col5:
    NHR = st.text_input('NHR', '0.0221')

  with col1:
    HNR = st.text_input('HNR', '21.03')
  with col2:
    RPDE = st.text_input('RPDE', '0.414')
  with col3:
    DFA = st.text_input('DFA', '0.815')
  with col4:
    spread1 = st.text_input('spread1', '-4.813')
  with col5:
    spread2 = st.text_input('spread2', '0.266')

  with col1:
    D2 = st.text_input('D2', '2.301')
  with col2:
    PPE = st.text_input('PPE', '0.284')

  if st.button("Parkinson's Test Result"):
    try:
      user_input = [
          float(fo),
          float(fhi),
          float(flo),
          float(Jitter_percent),
          float(Jitter_Abs),
          float(RAP),
          float(PPQ),
          float(DDP),
          float(Shimmer),
          float(Shimmer_dB),
          float(APQ3),
          float(APQ5),
          float(APQ),
          float(DDA),
          float(NHR),
          float(HNR),
          float(RPDE),
          float(DFA),
          float(spread1),
          float(spread2),
          float(D2),
          float(PPE),
      ]
      parkinsons_prediction = parkinsons_model.predict([user_input])

      if parkinsons_prediction[0] == 1:
        parkinsons_diagnosis = "The person has Parkinson's disease"
        st.error(parkinsons_diagnosis)
      else:
        parkinsons_diagnosis = "The person does not have Parkinson's disease"
        st.success(parkinsons_diagnosis)

      save_prediction('Parkinsons', parkinsons_diagnosis)
    except Exception as e:
      st.warning('Please enter valid numeric values in all fields.')


# ==========================================
# 4. Breast Cancer Prediction Page
# ==========================================
if selected == 'Breast Cancer Prediction':
  st.title('Breast Cancer Prediction using ML')
  st.markdown('### Please enter the medical parameters:')

  features = [
      'radius_mean',
      'texture_mean',
      'perimeter_mean',
      'area_mean',
      'smoothness_mean',
      'compactness_mean',
      'concavity_mean',
      'concave points_mean',
      'symmetry_mean',
      'fractal_dimension_mean',
      'radius_se',
      'texture_se',
      'perimeter_se',
      'area_se',
      'smoothness_se',
      'compactness_se',
      'concavity_se',
      'concave points_se',
      'symmetry_se',
      'fractal_dimension_se',
      'radius_worst',
      'texture_worst',
      'perimeter_worst',
      'area_worst',
      'smoothness_worst',
      'compactness_worst',
      'concavity_worst',
      'concave points_worst',
      'symmetry_worst',
      'fractal_dimension_worst',
  ]

  # Organize the 30 fields into 3 clean columns
  input_values = []
  cols = st.columns(3)
  for i, feature in enumerate(features):
    with cols[i % 3]:
      val = st.number_input(
          f"{feature.replace('_', ' ').title()}", value=1.0, format='%.5f'
      )
      input_values.append(val)

  if st.button('Breast Cancer Test Result'):
    cancer_prediction = breast_cancer_model.predict([input_values])

    if cancer_prediction[0] == 1:
      cancer_diagnosis = (
          'The model predicts that the patient has Breast Cancer (Malignant).'
      )
      st.error(cancer_diagnosis)
    else:
      cancer_diagnosis = (
          'The model predicts that the patient is Healthy (Benign).'
      )
      st.success(cancer_diagnosis)

    save_prediction('Breast Cancer', cancer_diagnosis)


# ==========================================
# 5. Patient Diagnostic Audit Records (Database)
# ==========================================
if selected == 'Patient History':
  st.title('Patient Diagnostic Audit Records (SQLite)')
  st.info(
      'Historical records of all diagnostic predictions generated by the ML'
      ' models.'
  )

  conn = sqlite3.connect('patient_records.db')
  df = pd.read_sql_query('SELECT * FROM history ORDER BY id DESC', conn)
  conn.close()

  if not df.empty:
    st.dataframe(df, use_container_width=True)
  else:
    st.write('No records found yet. Perform a test prediction first.')


# ==========================================
# Footer
# ==========================================
st.markdown(
    """
    <hr style="margin-top:50px;">
    <div style="text-align:center; color:gray;">
        Developed by <b>Sonu Nahak</b> 🧑‍💻
    </div>
    """,
    unsafe_allow_html=True,
)