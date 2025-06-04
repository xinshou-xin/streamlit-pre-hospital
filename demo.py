import streamlit as st
import joblib
from PIL import Image
import plotly.express as px
import pandas as pd
import base64
import io
import shap
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg") 

# Page configuration 🤖
st.set_page_config(page_title="Mortality Prediction", layout="wide", page_icon="🦈")

st.set_option('deprecation.showPyplotGlobalUse', False)
# Model selection
model_choice = st.sidebar.radio("Select Model", ["M1 (Pre-hospital)", "M2 (Post-hospital)"])

# ====================== Input: A Features (Prehospital) ======================
st.markdown("## Pre-hospital Return of Spontaneous Circulation") #(Prehospital ROSC)
a_data = {}
with st.expander("Pre-hospital Data", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        a_data["age"] = st.number_input("Age", 18, 120, 65)
        a_data["Bystander use of AEDs"] = int(st.selectbox("AED used by bystander", ["No", "Yes"]) == "Yes")
        a_data["Time to ambulance arrival"] = st.number_input("Ambulance arrival time (minutes)", 1, 60, 10)
        a_data["Performer of defibrillation_Medical staff"] = int(st.selectbox("Defibrillation performed by medical staff", ["No", "Yes"]) == "Yes")
        a_data["Bystander CPR"] = int(st.selectbox("Bystander CPR performed", ["No", "Yes"]) == "Yes")
        # a_data["Initial rhythm of cardiac arrest_Normal heart rhythm"] = int(st.selectbox("Initial rhythm: Normal", ["No", "Yes"]) == "Yes")
    with col2:
        # a_data["Initial rhythm of cardiac arrest_Shockable heart rhythm"] = int(st.selectbox("Initial rhythm: Shockable", ["No", "Yes"]) == "Yes")
        # a_data["Initial rhythm of cardiac arrest_Non-shockable heart rhythm"] = int(st.selectbox("Initial rhythm: Non-shockable", ["No", "Yes"]) == "Yes")
        initial_rhythm = st.selectbox(
            "Initial rhythm of cardiac arrest",
            ["Normal heart rhythm", "Shockable heart rhythm", "Non-shockable heart rhythm"]
        )

        a_data["Initial rhythm of cardiac arrest_Normal heart rhythm"] = int(initial_rhythm == "Normal heart rhythm")
        a_data["Initial rhythm of cardiac arrest_Shockable heart rhythm"] = int(initial_rhythm == "Shockable heart rhythm")
        a_data["Initial rhythm of cardiac arrest_Non-shockable heart rhythm"] = int(initial_rhythm == "Non-shockable heart rhythm")

        a_data["Out-of-hospital electrical defibrillation"] = int(st.selectbox("Out-of-hospital defibrillation", ["No", "Yes"]) == "Yes")
        a_data["Location_Family house"] = int(st.selectbox("Event occurred at family house", ["No", "Yes"]) == "Yes")
        a_data["Coronary heart disease-related factors present"] = int(st.selectbox("CHD-related factors present", ["No", "Yes"]) == "Yes")
        a_data["5-minute social rescue circle"] = int(st.selectbox("5-minute social rescue circle", ["No", "Yes"]) == "Yes")

# ====================== Input: B Features (Only for M2) ======================
b_data = {}
if model_choice == "M2 (Post-hospital)":
    st.markdown("## 30-day survival after hospital discharge ")
    with st.expander("Post-hospital Data", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            b_data["Use of electrical defibrillation in ED"] = int(st.selectbox("Defibrillation in ED", ["No", "Yes"]) == "Yes")
            b_data["Establishment of advanced artificial airway in ED"] = int(st.selectbox("Advanced airway in ED", ["No", "Yes"]) == "Yes")
            b_data["PCI for ED"] = int(st.selectbox("PCI in ED", ["No", "Yes"]) == "Yes")
            b_data["ECMO for ED"] = int(st.selectbox("ECMO in ED", ["No", "Yes"]) == "Yes")
            
        with col2:
            b_data["TTM for ED"] = int(st.selectbox("TTM in ED", ["No", "Yes"]) == "Yes")
            b_data["Use of mechanical CPR device in ED"] = int(st.selectbox("Mechanical CPR device in ED", ["No", "Yes"]) == "Yes")
            b_data["Use of medications in ED"] = int(st.selectbox("Medications used in ED", ["No", "Yes"]) == "Yes")
            b_data["Return of spontaneous circulation in ED"] = int(st.selectbox("ROSC in ED", ["No", "Yes"]) == "Yes")

# ====================== Prediction Section ======================

model1_path = "M1_compare/modelsM1/catboost_model_fold_2.pkl" 
model2_path = "M2_compare/modelsM2/catboost_model_fold_2.pkl" 

shap_fig1_path = "M1_compare/SHAP/shap_summary_plot.png"
shap_fig2_path = "M2_compare/SHAP/shap_summary_plot.png"

shap1_data_csv = pd.read_csv("M1_compare/SHAP/shap_data.csv")
shap1_value_csv = pd.read_csv("M1_compare/SHAP/shap_values.csv")
shap2_data_csv = pd.read_csv("M2_compare/SHAP/shap_data.csv")
shap2_value_csv = pd.read_csv("M2_compare/SHAP/shap_values.csv")

x_features_m1 = [
    'age','Bystander use of AEDs','Time to ambulance arrival','Performer of defibrillation_Medical staff',
    'Bystander CPR','Initial rhythm of cardiac arrest_Normal heart rhythm',
    'Initial rhythm of cardiac arrest_Shockable heart rhythm','Initial rhythm of cardiac arrest_Non-shockable heart rhythm',
    'Out-of-hospital electrical defibrillation','Location_Family house',
    'Coronary heart disease-related factors present','5-minute social rescue circle'
]
x_features_m2 = x_features_m1 + [
    'Use of electrical defibrillation in ED','Use of mechanical CPR device in ED',
    'Establishment of advanced artificial airway in ED','Use of medications in ED',
    'PCI for ED','TTM for ED','ECMO for ED','Return of spontaneous circulation in ED'
]

model = joblib.load(model1_path)
# === 默认：加载模型、特征、SHAP图路径（以 M1 为默认）
if model_choice == "M1 (Pre-hospital)":
    default_model = joblib.load(model1_path)
    default_features = x_features_m1
    shap_fig_path = shap_fig1_path
    shap_data = shap1_data_csv
    shap_value = shap1_value_csv
elif model_choice == "M2 (Post-hospital)":
    default_model = joblib.load(model2_path)
    default_features = x_features_m2
    shap_fig_path = shap_fig2_path
    shap_data = shap2_data_csv
    shap_value = shap2_value_csv

predict_clicked = st.button("🚀 Predict")

# ========== 图 1：Summary + Dependence ==========
# st.write("<h2>SHAP Analysis Visualization</h2>", unsafe_allow_html=True)
# col1, col2 = st.columns(2)
# with col1:
#     st.markdown('<h5 style="color: #0775eb">Summary plot:</h5>', unsafe_allow_html=True)
#     st.image(Image.open(shap_fig_path), caption=f"SHAP Summary Plot ({model_choice})")

# with col2:
#     st.markdown('<h5 style="color: #0775eb">Dependence plot:</h5>', unsafe_allow_html=True)
#     selected_feature = st.selectbox("Choose a feature", shap_data.columns)
#     fig = px.scatter(
#         x=shap_data[selected_feature],
#         y=shap_value[selected_feature],
#         color=shap_data[selected_feature],
#         color_continuous_scale=["blue", "red"],
#         labels={"x": "Original value", "y": "SHAP value"},
#     )
#     st.write(fig)

# ========== 图 2+3：Waterfall + Force，预测后才显示 ==========

if predict_clicked:

    # st.write(a_data)
    # st.write(b_data)
    if model_choice == "M1 (Pre-hospital)":
        model = joblib.load(model1_path)
        features = x_features_m1
        X_input = [a_data[feat] for feat in features]

    else:
        model = joblib.load(model2_path)
        full_data = {**a_data, **b_data}
        features = x_features_m2
        X_input = [full_data[feat] for feat in features]

    result = model.predict([X_input])
    pred = result[0]

    if model_choice == "M1 (Pre-hospital)":
        label = "Return of Spontaneous Circulation (ROSC)" if pred == 1 else "No ROSC"
    else:  # M2 model
        label = "Survived" if pred == 1 else "Deceased"

    st.write("<h2>Predict Result</h2>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='
        background-color: #e6f4ea;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #34a853;
        text-align: center;
        font-size: 20px;
        color: #202124;
        font-weight: bold;'>
        ✅ Prediction Result:  <span style='color: #0b8043'>{label}</span>
    </div>
    """, unsafe_allow_html=True)

    # =================== SHAP 图像区 ===================
    st.write("<h2>SHAP Analysis Visualization</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<h5 style="color: #0775eb">Summary plot:</h5>', unsafe_allow_html=True)
        st.image(Image.open(shap_fig_path), caption=f"SHAP Summary Plot ({model_choice})")
    with col2:
        st.markdown('<h5 style="color: #0775eb">Dependence plot:</h5>', unsafe_allow_html=True)
        selected_feature = st.selectbox("Choose a feature", shap_data.columns)
        fig = px.scatter(
            x=shap_data[selected_feature],
            y=shap_value[selected_feature],
            color=shap_data[selected_feature],
            color_continuous_scale=["blue", "red"],
            labels={"x": "Original value", "y": "SHAP value"},
        )
        st.write(fig)

    # ============ Waterfall ============
    st.write("<h2>Personalized Risk Interpretation</h2>", unsafe_allow_html=True)
    st.markdown('<h5 style="color: #0775eb">Waterfall plot:</h5>', unsafe_allow_html=True)
    explainer = shap.Explainer(model)
    X_input = pd.DataFrame([X_input], columns=features)
    shap_values = explainer(X_input)
    plt.figure(figsize=(8, 5))
    shap.waterfall_plot(shap_values[0], max_display=shap_values[0].values.shape[0], show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    st.markdown(
        f'<div style="display: flex; justify-content: center;">'
        f'<img src="data:image/png;base64,{base64.b64encode(buf.read()).decode()}" alt="Waterfall plot" style="width: 900px;">'
        f'</div>', unsafe_allow_html=True)
    plt.close()

    # ============ Force Plot ============
    st.markdown('<h5 style="color: #0775eb">Force plot:</h5>', unsafe_allow_html=True)
    shap_values_array = shap.Explainer(model).shap_values(X_input.values)
    fig = shap.force_plot(
        explainer.expected_value, shap_values_array[0], X_input, matplotlib=True
    )
    st.pyplot(fig)


st.sidebar.markdown("""
    ### ℹ️ Model Information

    **M1 (Pre-hospital):**  
    This model predicts whether the patient will achieve **Return of Spontaneous Circulation (ROSC)** at the scene based on pre-hospital features such as bystander actions, initial rhythm, and EMS response time.

    **M2 (Post-hospital):**  
    This model predicts **in-hospital survival** using both pre-hospital and in-hospital features, including interventions during resuscitation and early hospital care.
    """)