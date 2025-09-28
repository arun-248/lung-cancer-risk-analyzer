# App.py

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import matplotlib.pyplot as plt
import subprocess

st.set_page_config(page_title="Lung Cancer Risk Analyzer", page_icon="🫁", layout="wide")

DATA_PATH = Path("cancer patient data sets.csv")
MODEL_PATH = Path("model.pkl")
META_PATH = Path("model_meta.json")
IMAGE_PATHS = [Path("cancerimage.jpg")]
CM_PATH = Path("confusion_matrix.png")

@st.cache_data
def load_data(path: Path):
    return pd.read_csv(path)

@st.cache_resource
def load_model_and_meta():
    # 🔹 Auto-train fallback if model files are missing
    if not MODEL_PATH.exists() or not META_PATH.exists():
        with st.spinner("⚙️ Training model (first-time setup)..."):
            subprocess.run(["python", "train_model.py"], check=True)
    model = joblib.load(MODEL_PATH)
    meta = {}
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text())
    return model, meta

if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "🏠 Home"

tab_labels = ["🏠 Home", "🔮 Prediction", "📂 Case Study"]
tab_objects = st.tabs([f"**{label}**" for label in tab_labels])

with tab_objects[0]:
    col1, col2 = st.columns([0.65, 0.35])
    with col1:
        st.title("🫁 Lung Cancer Risk Analyzer")
        st.markdown(
            """
            ### Project Overview
            This interactive web application is designed to:
            - ✅ Predict **lung cancer risk level** (Low / Medium / High) from lifestyle and symptoms  
            - ✅ Provide clear **visual insights** into the dataset and model performance  
            - ✅ Demonstrate my **skills in data science, ML, and UI/UX design**  
            - ✅ Showcase **end-to-end machine learning deployment** using Streamlit and Python  
            - ✅ Highlight my ability to build **practical, user-friendly health tech applications**  
            - ✅ Emphasize the importance of **data-driven decision making in healthcare**  
            """
        )
        if META_PATH.exists():
            meta = json.loads(META_PATH.read_text())
            st.success(f"Current model: **{meta['best_model']}** | F1_macro: **{meta['f1_macro']:.3f}**")
        if st.button("🧪 Start Prediction", use_container_width=True, type="primary"):
            js = """
            <script>
            var tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
            tabs[1].click();
            </script>
            """
            st.components.v1.html(js, height=0)
    with col2:
        for img_path in IMAGE_PATHS:
            if img_path.exists():
                st.image(str(img_path), use_container_width=True, output_format="PNG")
                st.markdown(
                    """
                    <style>
                    img {
                        max-height: 75% !important;
                        object-fit: contain;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

with tab_objects[1]:
    model, meta = load_model_and_meta()
    st.subheader("🔮 Predict Risk Level")
    st.caption("Fill in the form below and click **Predict**.")
    with st.form("pred_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 1, 120, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            air = st.slider("Air Pollution", 1, 8, 3)
            alcohol = st.slider("Alcohol use", 1, 8, 2)
            dust = st.slider("Dust Allergy", 1, 8, 3)
            occ = st.slider("Occupational Hazards", 1, 8, 3)
            genetic = st.slider("Genetic Risk", 1, 8, 3)
            cld = st.slider("Chronic Lung Disease", 1, 8, 2)
        with col2:
            diet = st.slider("Balanced Diet", 1, 8, 4)
            obesity = st.slider("Obesity", 1, 8, 3)
            smoking = st.slider("Smoking", 1, 8, 2)
            passive = st.slider("Passive Smoker", 1, 8, 2)
            chest = st.slider("Chest Pain", 1, 8, 2)
            blood = st.slider("Coughing of Blood", 1, 8, 1)
            fatigue = st.slider("Fatigue", 1, 8, 2)
            weight = st.slider("Weight Loss", 1, 8, 2)
        with col3:
            breath = st.slider("Shortness of Breath", 1, 8, 2)
            wheeze = st.slider("Wheezing", 1, 8, 2)
            swallow = st.slider("Swallowing Difficulty", 1, 8, 2)
            club = st.slider("Clubbing of Finger Nails", 1, 8, 1)
            cold = st.slider("Frequent Cold", 1, 8, 2)
            dry = st.slider("Dry Cough", 1, 8, 2)
            snore = st.slider("Snoring", 1, 8, 2)
        submitted = st.form_submit_button("Predict")
    if submitted:
        gender_num = 1 if gender == "Male" else 2
        row = pd.DataFrame([{
            'Age': age, 'Gender': gender_num,
            'Air Pollution': air, 'Alcohol use': alcohol, 'Dust Allergy': dust,
            'OccuPational Hazards': occ, 'Genetic Risk': genetic, 'chronic Lung Disease': cld,
            'Balanced Diet': diet, 'Obesity': obesity, 'Smoking': smoking, 'Passive Smoker': passive,
            'Chest Pain': chest, 'Coughing of Blood': blood, 'Fatigue': fatigue, 'Weight Loss': weight,
            'Shortness of Breath': breath, 'Wheezing': wheeze, 'Swallowing Difficulty': swallow,
            'Clubbing of Finger Nails': club, 'Frequent Cold': cold, 'Dry Cough': dry, 'Snoring': snore
        }])
        pred = model.predict(row)[0]
        color_map = {"Low": "#16a34a", "Medium": "#f59e0b", "High": "#dc2626"}
        color = color_map.get(pred, "#3b82f6")
        st.markdown(
            f"""
            <div style='padding:1rem;border-radius:0.75rem;background:{color}20;border:1px solid {color};'>
                <h3 style='margin:0;'>Predicted Risk: <span style='color:{color}'>{pred}</span></h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(row)[0]
            classes = model.classes_ if hasattr(model, "classes_") else meta.get("classes", [])
            st.markdown("### Class Probabilities")
            for cls, p in zip(classes, proba):
                st.progress(float(p))
                st.write(f"**{cls}**: {p:.2%}")
        st.markdown("### Recommendation")
        if pred == "High":
            st.error("⚠️ High risk detected. Please consult a doctor immediately.")
        elif pred == "Medium":
            st.warning("🟡 Moderate risk. Consider medical screening and lifestyle changes.")
        else:
            st.success("🟢 Low risk. Maintain healthy habits and regular check-ups.")

with tab_objects[2]:
    st.title("📂 Case Study")
    st.header("📊 Dataset Overview")
    if DATA_PATH.exists():
        df = load_data(DATA_PATH)
        st.write(df.head())
        st.caption(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
        col1, col2 = st.columns(2)
        with col1:
            if "Age" in df.columns:
                st.markdown("**Age Distribution**")
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.hist(df["Age"], bins=20, color="lightblue", edgecolor="black")
                ax.set_xlabel("Age")
                ax.set_ylabel("Count")
                st.pyplot(fig, clear_figure=True)
        with col2:
            if "Level" in df.columns:
                st.markdown("**Risk Level Distribution**")
                fig2, ax2 = plt.subplots(figsize=(4, 3))
                df["Level"].value_counts().plot(kind="bar", ax=ax2, color="teal")
                ax2.set_xlabel("Risk Level")
                ax2.set_ylabel("Count")
                st.pyplot(fig2, clear_figure=True)
    st.header("⚙️ Methodology")
    st.write(
        """
        - Data preprocessing: encoding categorical features, scaling numerical values  
        - Feature engineering: combining lifestyle and symptom indicators into features  
        - Model training: Logistic Regression and Random Forest with GridSearchCV  
        - Model evaluation using macro F1 score and confusion matrix  
        """
    )
    st.header("🤖 Model Insights")
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text())
        st.success(f"Best model: **{meta['best_model']}** | F1_macro = {meta['f1_macro']:.3f}")
    if CM_PATH.exists():
        st.image(str(CM_PATH), caption="Confusion Matrix", width=400)
    st.header("📈 Key Findings & Highlights")
    st.write(
        """
        - Achieved macro F1 score close to **1.0** (due to synthetic dataset)  
        - Symptoms like **Smoking, Chronic Lung Disease, Fatigue, and Coughing of Blood** strongly correlated with high risk  
        - Visualization confirms distinct patterns between Low, Medium, and High risk groups  
        """
    )
    st.header("💡 Applications in Healthcare")
    st.write(
        """
        - Assists doctors in **early risk assessment** of patients  
        - Can be adapted for **preventive screening** and awareness campaigns  
        - Provides interpretable insights for both patients and practitioners  
        """
    )
    st.header("⚠️ Limitations")
    st.write(
        """
        - Dataset is synthetic → not validated for clinical use  
        - High accuracy may not generalize to real-world scenarios  
        - Requires extensive validation with real patient data  
        """
    )
    st.header("🚀 Future Enhancements")
    st.write(
        """
        - Integrate **larger and real-world datasets**  
        - Explore **XGBoost and Deep Learning** architectures  
        - Add **patient history tracking** with authentication  
        - Deploy on scalable platforms (Heroku, AWS, Streamlit Cloud)  
        """
    )
    st.header("📂 Conclusion")
    st.write(
        """
        This project demonstrates an **end-to-end machine learning pipeline** applied to healthcare.  
        While based on synthetic data, it highlights the **potential real-world impact** of AI-driven early risk assessment.  
        """
    )

st.markdown("---")
st.markdown(
    """
    ### 🚀 Project Developed By  
    **Arun Chinthalapally**
    """
)
st.caption("Built with ❤️ using Python & Streamlit")
