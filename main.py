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
from streamlit_echarts import st_echarts

matplotlib.use("Agg")
st.set_page_config(page_title="Mortality Prediction", layout="wide", page_icon="🦈")
st.set_option('deprecation.showPyplotGlobalUse', False)

# =================== Sidebar ===================
model_choice = st.sidebar.radio("Select Model", ["Model 1 (ROSC on-site)", "Model 2 (30-day survival)"])

# =================== Sidebar Info ===================
st.sidebar.markdown("""
### ℹ️ Model Background

**Model 1: On-site Return of Spontaneous Circulation (ROSC)**  
Developed to support real-time pre-hospital decision-making, this model predicts the probability of achieving ROSC following out-of-hospital cardiac arrest (OHCA). It is based on early-phase variables including witnessed status, bystander CPR/AED application, initial cardiac rhythm, and EMS response intervals.

**Model 2: 30-day Survival Post-OHCA**  
This model estimates the likelihood of 30-day survival following OHCA by integrating both pre-hospital and in-hospital clinical factors. These include resuscitative efforts, airway and circulatory interventions, and early hospital management, offering insight into short-term prognosis.
""")


# 更改预测状态按钮
# if st.sidebar.button("🔄 Reset Prediction"):
#     st.session_state["predict_done"] = False

# =================== Header(免责声明) ===================
st.markdown("""
<div style='
    background-color: #fff3cd;
    border-left: 6px solid #ffeeba;
    padding: 12px;
    border-radius: 8px;
    font-size: 16px;
    color: #856404;
    margin-bottom: 16px;
'>
    ⚠️ This web page is for testing purposes only. The data provided is for reference only and has no clinical significance.
</div>
""", unsafe_allow_html=True)

if model_choice == "Model 1 (ROSC on-site)":
    st.markdown("## Return of Spontaneous Circulation on-site")
elif model_choice == "Model 2 (30-day survival)":
    st.markdown("## 30-day survival")

# =================== Pre-hospital Features (A组) ===================
# st.markdown("## Return of Spontaneous Circulation on-site")
a_data = {}
with st.expander("pre-hospital data", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        a_data["age"] = st.number_input("Age", 18, 120, 65)
        a_data["Bystander use of AEDs"] = int(st.selectbox("AED used by bystander", ["No", "Yes"]) == "Yes")
        a_data["Time to ambulance arrival"] = st.number_input("Ambulance arrival time (minutes)", 1, 60, 10)
        a_data["Performer of defibrillation_Medical staff"] = int(st.selectbox("Defibrillation by medical staff", ["No", "Yes"]) == "Yes")
        a_data["Bystander CPR"] = int(st.selectbox("Bystander CPR performed", ["No", "Yes"]) == "Yes")
    with col2:
        initial_rhythm = st.selectbox("Initial rhythm of cardiac arrest", ["Normal heart rhythm", "Shockable heart rhythm", "Non-shockable heart rhythm"])
        a_data["Initial rhythm of cardiac arrest_Normal heart rhythm"] = int(initial_rhythm == "Normal heart rhythm")
        a_data["Initial rhythm of cardiac arrest_Shockable heart rhythm"] = int(initial_rhythm == "Shockable heart rhythm")
        a_data["Initial rhythm of cardiac arrest_Non-shockable heart rhythm"] = int(initial_rhythm == "Non-shockable heart rhythm")
        a_data["Out-of-hospital electrical defibrillation"] = int(st.selectbox("Out-of-hospital defibrillation", ["No", "Yes"]) == "Yes")
        a_data["Location_Family house"] = int(st.selectbox("Event occurred at family house", ["No", "Yes"]) == "Yes")
        # a_data["Coronary heart disease-related factors present"] = int(st.selectbox("CHD-related factors present", ["No", "Yes"]) == "Yes")
        a_data["5-minute social rescue circle"] = int(st.selectbox("Construction of the 5-minute social rescue circle", ["No", "Yes"]) == "Yes")

# =================== Post-hospital Features (B组，仅M2) ===================
b_data = {}
if model_choice == "Model 2 (30-day survival)":
    # st.markdown("## 30-day survival")
    with st.expander("in-hospital data", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            b_data["Use of electrical defibrillation in ED"] = int(st.selectbox("Defibrillation in ED", ["No", "Yes"]) == "Yes")
            b_data["Establishment of advanced artificial airway in ED"] = int(st.selectbox("Advanced airway in ED", ["No", "Yes"]) == "Yes")
            b_data["PCI for ED"] = int(st.selectbox("Percutaneous coronary intervention (PCI) for ED", ["No", "Yes"]) == "Yes")
            b_data["ECMO for ED"] = int(st.selectbox("Extracorporeal membrane oxygenation (ECMO) for ED", ["No", "Yes"]) == "Yes")
        with col2:
            b_data["TTM for ED"] = int(st.selectbox("Therapeutic temperature management (TTM) for ED", ["No", "Yes"]) == "Yes")
            b_data["Use of mechanical CPR device in ED"] = int(st.selectbox("Mechanical CPR device in ED", ["No", "Yes"]) == "Yes")
            b_data["Use of medications in ED"] = int(st.selectbox("Medications used in ED", ["No", "Yes"]) == "Yes")
            b_data["Return of spontaneous circulation in ED"] = int(st.selectbox("Return of spontaneous circulation in ED", ["No", "Yes"]) == "Yes")

# =================== Load Models & Data ===================
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
    'Out-of-hospital electrical defibrillation','Location_Family house','5-minute social rescue circle'
]# 'Coronary heart disease-related factors present'
x_features_m2 = x_features_m1 + [
    'Use of electrical defibrillation in ED','Use of mechanical CPR device in ED',
    'Establishment of advanced artificial airway in ED','Use of medications in ED',
    'PCI for ED','TTM for ED','ECMO for ED','Return of spontaneous circulation in ED'
]

if model_choice == "Model 1 (ROSC on-site)":
    model = joblib.load(model1_path)
    shap_fig_path = shap_fig1_path
    shap_data = shap1_data_csv
    shap_value = shap1_value_csv
    features = x_features_m1
else:
    model = joblib.load(model2_path)
    shap_fig_path = shap_fig2_path
    shap_data = shap2_data_csv
    shap_value = shap2_value_csv
    features = x_features_m2

# =================== Prediction ===================
# 按钮样式
st.markdown("""
    <style>
        /* 默认样式 */
        .stButton > button {
            display: block;
            margin: 0 auto;
            background-color: #e8f0fe;
            color: #1967d2;
            border: 1px solid #1967d2;
            border-radius: 10px;
            padding: 12px 28px;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        /* hover 状态 */
        .stButton > button:hover {
            background-color: #d2e3fc;
            color: #174ea6;
            border-color: #174ea6;
        }

        /* 按下状态 */
        .stButton > button:active {
            background-color: #e6f4ea;  /* 淡绿色 */
            color: #137333;
            border-color: #137333;
        }

        /* 点击后获得焦点状态 */
        .stButton > button:focus:not(:active) {
            background-color: #e6f4ea;
            color: #137333;
            border-color: #137333;
        }
    </style>
""", unsafe_allow_html=True)
if st.button("🚀 Predict"):
    st.session_state["predict_done"] = True
# 仪表盘配置模板函数
def get_gauge_option(value):
    return {
        "series": [
            {
                "type": "gauge",
                "center": ["50%", "60%"],
                "startAngle": 200,
                "endAngle": -20,
                "min": 0,
                "max": 100,
                "splitNumber": 10,
                "itemStyle": {
                    "color": "#91cc75"
                },
                "progress": {
                    "show": True,
                    "width": 30
                },
                "pointer": {
                    "show": False
                },
                "axisLine": {
                    "lineStyle": {
                        "width": 30
                    }
                },
                "axisTick": {
                    "distance": -45,
                    "splitNumber": 5,
                    "lineStyle": {
                        "width": 2,
                        "color": "#999"
                    }
                },
                "splitLine": {
                    "distance": -52,
                    "length": 14,
                    "lineStyle": {
                        "width": 3,
                        "color": "#999"
                    }
                },
                "axisLabel": {
                    "distance": -20,
                    "color": "#666",
                    "fontSize": 14
                },
                "anchor": {
                    "show": False
                },
                "title": {
                    "show": False  # 隐藏内部 title
                },
                "detail": {
                    "valueAnimation": True,
                    "width": "60%",
                    "lineHeight": 40,
                    "borderRadius": 8,
                    "offsetCenter": [0, "-15%"],
                    "fontSize": 30,
                    "fontWeight": "bolder",
                    "formatter": "{value} %",
                    "color": "#000"
                },
                "data": [
                    {
                        "value": round(value * 100, 1)
                    }
                ]
            }
        ]
    }


if st.session_state.get("predict_done", False):
    if model_choice == "Model 1 (ROSC on-site)":
        X_input = [a_data[feat] for feat in features]
        proba = model.predict_proba([X_input])[0][1]
        label_text = "Probability of ROSC"
    else:
        full_data = {**a_data, **b_data}
        X_input = [full_data[feat] for feat in features]
        proba = model.predict_proba([X_input])[0][1]
        label_text = "Probability of Survival"
    st.write("<h2>Predict Result</h2>", unsafe_allow_html=True)
    
    # 显示标题
    # st.markdown(f"""
    # <div style='
    #     text-align: left;
    #     font-size: 22px;
    #     font-weight: bold;
    #     margin-bottom: 10px;
    #     color: #202124'>
    #     ✅ {label_text}
    # </div>
    # """, unsafe_allow_html=True)
    st.markdown(f'<h5 style="color: #0775eb"> {label_text}:</h5>', unsafe_allow_html=True)

    # 显示仪表盘
    st_echarts(get_gauge_option(proba), height="400px")

    # =================== SHAP 可视化 ===================
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

    # =================== Waterfall ===================
    st.write("<h2>Personalized Risk Interpretation</h2>", unsafe_allow_html=True)
    st.markdown('<h5 style="color: #0775eb">Waterfall plot:</h5>', unsafe_allow_html=True)
    explainer = shap.Explainer(model)
    X_input_df = pd.DataFrame([X_input], columns=features)
    shap_values = explainer(X_input_df)
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

    # =================== Force ===================
    st.markdown('<h5 style="color: #0775eb">Force plot:</h5>', unsafe_allow_html=True)
    shap_values_array = shap.Explainer(model).shap_values(X_input_df.values)
    fig = shap.force_plot(explainer.expected_value, shap_values_array[0], X_input_df, matplotlib=True)
    st.pyplot(fig)


