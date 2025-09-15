from google.colab import drive
drive.mount('/content/drive')

# ---- Load model and dataset ----
model = joblib.load('/content/drive/MyDrive/Datasets/heart_dataset/heart_model.pkl')
df = pd.read_csv("/content/drive/MyDrive/Datasets/heart_dataset/heart.csv")

# ---- Page config ----
st.title("🫀 Heart Disease Risk Predictor")
st.markdown("Enter patient data in the sidebar and click Predict to see risk probability and analytics.")

# ---- Sidebar input form ----
st.sidebar.header("Patient Data Input")
age = st.sidebar.number_input("Age", 1, 120, 50)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
cp = st.sidebar.selectbox("Chest Pain Type", ["typical angina","atypical angina","non-anginal","asymptomatic"])
trestbps = st.sidebar.number_input("Resting Blood Pressure", 50, 250, 120)
chol = st.sidebar.number_input("Cholesterol", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
restecg = st.sidebar.selectbox("Resting ECG", ["normal","st-t abnormality","lv hypertrophy"])
thalch = st.sidebar.number_input("Max Heart Rate Achieved", 60, 250, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.sidebar.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
slope = st.sidebar.selectbox("Slope of ST segment", ["upsloping","flat","downsloping"])
ca = st.sidebar.number_input("Number of Major Vessels Colored", 0, 4, 0)
thal = st.sidebar.selectbox("Thalassemia", ["normal","fixed defect","reversible defect"])

# ---- Map categorical to numeric ----
sex_map = {"Male":1, "Female":0}
cp_map = {"typical angina":0,"atypical angina":1,"non-anginal":2,"asymptomatic":3}
restecg_map = {"normal":0,"st-t abnormality":1,"lv hypertrophy":2}
slope_map = {"upsloping":0,"flat":1,"downsloping":2}
thal_map = {"normal":1,"fixed defect":2,"reversible defect":3}
fbs_map = {"Yes":1,"No":0}
exang_map = {"Yes":1,"No":0}

# ---- Predict button ----
if st.sidebar.button("Predict"):
    # Create input dataframe
    input_data = pd.DataFrame({
        'age':[age],
        'sex':[sex_map[sex]],
        'cp':[cp_map[cp]],
        'trestbps':[trestbps],
        'chol':[chol],
        'fbs':[fbs_map[fbs]],
        'restecg':[restecg_map[restecg]],
        'thalch':[thalch],
        'exang':[exang_map[exang]],
        'oldpeak':[oldpeak],
        'slope':[slope_map[slope]],
        'ca':[ca],
        'thal':[thal_map[thal]]
    })

    # ---- Predict ----
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # ---- Top row: Probability gauge ----
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.subheader("🫀 Heart Disease Risk")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range':[0,33], 'color':'green'},
                    {'range':[33,66], 'color':'orange'},
                    {'range':[66,100], 'color':'red'}
                ],
                'threshold': {'line': {'color': "black", 'width': 4}, 'value': prob*100}
            }
        ))
        fig.update_layout(height=300, margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

        # Risk label
        if prob > 0.7:
            st.markdown(f'<p style="color:red;font-size:24px">⚠️ High Risk ({prob:.2f})</p>', unsafe_allow_html=True)
        elif prob > 0.4:
            st.markdown(f'<p style="color:orange;font-size:24px">⚠️ Moderate Risk ({prob:.2f})</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p style="color:green;font-size:24px">🫀 Low Risk ({prob:.2f})</p>', unsafe_allow_html=True)

    # ---- Middle row: Simple Feature Analysis ----
    st.subheader("📊 Feature Values Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Input Feature Values**")
        st.bar_chart(input_data.T.rename(columns={0:"Value"}))

    with col2:
        st.markdown("**EDA Visuals**")
        st.markdown("Cholesterol Trend")
        st.line_chart(df['chol'])
        st.markdown("ST Depression Distribution")
        st.bar_chart(df['oldpeak'])

    # ---- Bottom row: Input overview ----
    st.subheader("🩺 Input Data Overview")
    st.dataframe(input_data.T.rename(columns={0:"Value"}))

# ---- Footer ----
st.markdown("---")

