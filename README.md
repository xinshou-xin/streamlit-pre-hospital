
# 🚑 Pre-Hospital ROSC & 30-Day Survival Prediction App

**Streamlit-based clinical AI tool** that predicts:

* **Model 1 (ROSC)**: Return of Spontaneous Circulation at the scene
* **Model 2 (30-Day Survival)**: Survival within 30 days post-OHCA

Designed for **real-time decision support** in out-of-hospital cardiac arrest (OHCA), this project integrates machine learning with clinical domain knowledge to assist pre-hospital and emergency care providers.

---

## 📘 Table of Contents

* [🔍 Overview](#-overview)
* [🧠 Models Description](#-models-description)
* [🖥️ App Features](#️-app-features)
* [📦 Installation & Deployment](#-installation--deployment)
* [🧪 Dataset & Preprocessing](#-dataset--preprocessing)
* [⚠️ Disclaimer](#️-disclaimer)
* [📬 Contact](#-contact)

---

## 🔍 Overview

Out-of-hospital cardiac arrest (OHCA) is a time-sensitive emergency. Prognostication tools can guide clinical decisions, resource allocation, and communication. This app provides real-time probability estimates of:

* ROSC (Model 1): Whether spontaneous circulation returns on site
* 30-Day Survival (Model 2): Whether the patient survives 30 days post-OHCA

Both models are based on **CatBoost classifiers**, trained on labeled registry data, and made interpretable with **SHAP** visualizations.

---

## 🧠 Models Description

### 🩺 Model 1 – On-site ROSC Prediction

* **Goal**: Estimate the likelihood of successful resuscitation before hospital arrival.
* **Inputs**:

  * Witnessed status
  * Bystander CPR/AED
  * Initial rhythm
  * EMS arrival times
  * Prehospital interventions (airway, adrenaline, etc.)
* **Output**: Probability of ROSC at the scene.

### 🏥 Model 2 – 30-Day Survival Prediction

* **Goal**: Estimate survival probability 30 days after OHCA.
* **Inputs**:

  * All Model 1 features
  * Hospital-based interventions
  * ROSC status at ED
  * Transport and ED timing data
* **Output**: Estimated 30-day survival chance.

---

## 🖥️ App Features

✅ **Interactive User Input**:
Easily enter model variables using dropdowns, sliders, and forms.

✅ **Dual Model Switching**:
Choose between ROSC or survival prediction via sidebar toggle.

✅ **Visual Model Explanation**:
Understand model output with SHAP bar plots showing feature contributions.

✅ **Persistent State**:
Predictions are updated only when the “Predict” button is clicked to avoid auto-refresh.

✅ **Compact UI**:
Tabbed or accordion-style layout to group features by phase (pre-hospital vs in-hospital).

---



## 📦 Installation & Deployment

### 🔧 Local Setup

```bash
# Clone the repo
git clone https://github.com/xinshou-xin/streamlit-pre-hospital.git
cd OHCA

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Launch the app
streamlit run main.py
```

### ☁️ Streamlit Cloud

The app is deployable via [Streamlit Community Cloud](https://streamlit.io/cloud). Just connect your GitHub repo, and make sure you include:

* `main.py` (entry point)
* `requirements.txt`
* `.streamlit/runtime.txt` (e.g. `python-3.10`)

---


## 🧪 Dataset & Preprocessing

The models were trained using a structured OHCA registry with features including:

* Demographics: Age, location
* Prehospital events: CPR, defibrillation, time intervals
* ED-level interventions
* Outcomes: ROSC, survival

Missing values were handled with median/mode imputation. Feature importance was assessed using SHAP values and domain knowledge.

---

## ⚠️ Disclaimer

> This tool is intended **for research and educational use only**.
> It is **not approved** for clinical use and should **not replace medical judgment**.
> External validation and expert oversight are required before integration into EMS workflows.

---

## 📬 Contact

For collaboration, questions, or feedback:

**huangjinxin**
Email: `2602535898@qq.com`
GitHub: https://github.com/xinshou-xin

---
