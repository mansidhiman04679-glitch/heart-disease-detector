import streamlit as st

import pandas as pd
import numpy as np
import pickle
import base64


def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download=predictions.csv">Download Predictions CSV</a>'
    return href


st.title("Heart Disease Predictor")
tab1, tab2, tab3 = st.tabs(['Predict', 'Bulk Predict', 'Model Information'])

with tab1:
    age = st.number_input("Age (years)", min_value=0, max_value=150)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type",
                              ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG Results",
                               ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angine = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downloping"])

    # Convert categorial inputs to numeric
    sex = 0 if sex == "Male" else 1
    chest_pain = ["Atypical Angina", "Non-Anginal Pain", "Asymptomatic", "Typical Angina"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angine == "Yes" else 0
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    # Create  a Dataframe with user inputs
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })

    algonames = ['Decision Trees', 'Logistic Regression', 'Random Forest', 'Support Vector Machine', 'GridRandom']
    modelnames = ['DecisionTree.pkl', 'LogisticR.pkl', 'RandomForest.pkl', 'SVC.pkl', 'gridrf.pkl']


    def predict_heart_disease(data):
        predictions = []
        modelnames = ['LogisticR.pkl', 'DecisionTree.pkl', 'RandomForest.pkl', 'SVC.pkl', 'gridrf.pkl']
        for modelname in modelnames:
            with open(modelname, 'rb') as f:
                model = pickle.load(f)
            try:
                prediction = model.predict(data)
                predictions.append(prediction)
            except Exception as e:
                print(f"Error predicting with {modelname}: {e}")
                pass
        return predictions


    # create a submit button to make predictions
    if st.button("Submit"):
        st.subheader("Results.....")
        st.markdown('')

        result = predict_heart_disease(input_data)

        for i in range(len(result)):
            st.subheader(algonames[i])

            if result[i][0] == 0:
                st.write("No heart disease detected.")
            else:
                st.write("Heart disease detected.")
            st.markdown('')

with tab2:
    st.title("Upload CSV File")

    st.subheader('Instructions to note before uploading the file:')
    st.info("""
            1. No NaN values allowed.
            2. Total 11 features in this order ('Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS','RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope').\n
            3. Check the spellings of the feature names.
            4. Feature values conventions: \n
                - Age: age of the patient [years] \n
                - Sex: sex of the patient [0: Male, 1: Female] \n
                - ChestPainType: chest pain type [3: Typical Angina, 0: Atypical Angina, 1: Non-Anginal Pain, 2: Asymptomatic] \n
                - RestingBP: resting blood pressure [mm Hg] \n
                - Cholesterol: serum cholesterol [mm/dl] \n
                - FastingBS: fasting blood sugar [1: if FastingsBS > 120 mg/dl, 0: otherwise] \n
                - RestingECG: resting electrocardiogram results [0: Normal, 1: having ST-T wave abnormality (T wave inversingly)] \n
                - MaxHR: maximum heart rate achieved [ Numeric value between 60 and 202] \n
                - ExerciseAngina: exercise-induced angina [1: Yes, 0: No] \n
                - Oldpeak: oldpeak = ST [Numeric value measured in depression] \n
                - ST_Slope: the slope of the peak excercise ST segment [0: upsloping, 1: flat, 2: downsloping] \n
""")
    # create a file uploader to the side bar
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        model = pickle.load(open('LogisticR.pkl', 'rb'))

        expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG',
                            'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

        if set(expected_columns).issubset(input_data.columns):
            input_data['Prediction LR'] = ''

            for i in range(len(input_data)):
                arr = input_data.iloc[i, :-1].values
                input_data['Prediction LR'][i] = model.predict([arr])[0]
            input_data.to_csv('PredictedHeart.csv')

            st.subheader("Predictions:")
            st.write(input_data)

            # display the predictions
            st.subheader("Predictions:")
            st.write(input_data)

            st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)
        else:
            st.warning("Please make sure the upload CSV file has the correct column.")

    else:
        st.info("Upload a CSV file to get predictions.")

with tab3:
    import plotly.express as px

    data = {'Decision Tree': 80.97, 'Logistic Regression': 85.86, 'Random Forest': 84.23,
            'Support Vector Machine': 89.75}
    Models = list(data.keys())
    Accuracies = list(data.values())
    df = pd.DataFrame(list(zip(Models, Accuracies)), columns=['Models', 'Accuracies'])
    fig = px.bar(df, y='Accuracies', x='Models')
    st.plotly_chart(fig)

    import streamlit as st

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="centered"
)

st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

/* Main title */
h1 {
    color: #ffffff;
    text-align: center;
    font-weight: 700;
}

/* Sub-tabs */
.stTabs [data-baseweb="tab"] {
    font-size: 16px;
    font-weight: 600;
    color: #e0e0e0;
}

/* Input labels */
label {
    color: #cfd8dc !important;
    font-weight: 600;
}

/* Input boxes */
.stNumberInput input,
.stSelectbox div,
.stTextInput input {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 10px;
}

/* Buttons */
.stButton button {
    background: linear-gradient(90deg, #e53935, #e35d5b);
    color: white;
    font-size: 18px;
    border-radius: 12px;
    padding: 10px 25px;
    font-weight: bold;
}

.stButton button:hover {
    background: linear-gradient(90deg, #b71c1c, #d32f2f);
}

/* Cards */
.card {
    background-color: #111827;
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 20px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🫀 Heart Disease Prediction System</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#cfd8dc;'>Clinical Decision Support Tool</p>",
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)
age = st.number_input("Age (years)", 1, 120, 45, key="age")

sex = st.selectbox(
    "Sex",
    ["Male", "Female"],
    key="sex"
)

cp = st.selectbox(
    "Chest Pain Type",
    ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
    key="cp"
)

fbs = st.selectbox(
    "Fasting Blood Sugar",
    ["<= 120 mg/dl", "> 120 mg/dl"],
    key="fbs"
)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Predict Heart Disease Risk"):
    st.markdown("""
    <div class='card'>
        <h3 style='color:#ff5252;'>⚠️ High Risk Detected</h3>
        <p style='color:#cfd8dc;'>
        Please consult a cardiologist for further clinical evaluation.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    """
    <div style="
        background-color:#1f2a36;
        padding:20px;
        border-radius:15px;
        box-shadow:0 0 10px rgba(0,0,0,0.4);
    ">
    <h3>🩺 Patient Health Information</h3>
    </div>
    """,
    unsafe_allow_html=True
)





