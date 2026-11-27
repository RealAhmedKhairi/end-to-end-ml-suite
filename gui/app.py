import sys
from pathlib import Path

import cv2 as cv
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Get the main project folder path so files work from anywhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Allow importing clean_text from the text/ folder
sys.path.append(str(PROJECT_ROOT / "text"))
from preprocessing import clean_text


# Basic page title and layout
st.set_page_config(page_title="End-to-End ML Suite", layout="centered")
st.title("End-to-End Machine Learning Suite")
st.write("Choose a tab below to test our trained models on numerical, image, or text data.")


# @st.cache_resource loads the saved models ONCE and keeps them in memory
# instead of reloading them from disk every time you click a button.
@st.cache_resource
def load_saved_models():
    loaded = {
        # Part A: Numerical models & scaler
        "tabular_scaler": joblib.load(MODELS_DIR / "tabular_scaler.joblib"),
        "num_lr": joblib.load(MODELS_DIR / "linear_regression_model.joblib"),
        "num_knn": joblib.load(MODELS_DIR / "knn_model.joblib"),
        # Part B: Image models, scaler & label encoder
        "image_scaler": joblib.load(MODELS_DIR / "image_scaler.joblib"),
        "image_le": joblib.load(MODELS_DIR / "image_label_encoder.joblib"),
        # Part C: Text vectorizer & models
        "text_tfidf": joblib.load(MODELS_DIR / "text_tfidf.joblib"),
        "text_lr": joblib.load(MODELS_DIR / "text_lr_model.joblib"),
    }

    img_lr_path = MODELS_DIR / "image_lr_model.joblib"
    if img_lr_path.exists():
        loaded["img_lr"] = joblib.load(img_lr_path)

    # Image KNN is ignored in git due to file size, load it if generated locally
    img_knn_path = MODELS_DIR / "image_knn_model.joblib"
    if img_knn_path.exists():
        loaded["img_knn"] = joblib.load(img_knn_path)

    # Optional: load Linear SVM if saved in models/
    svm_path = MODELS_DIR / "text_svm_model.joblib"
    if svm_path.exists():
        loaded["text_svm"] = joblib.load(svm_path)

    return loaded


models = load_saved_models()

# Create 3 tabs matching the 3 parts of the project
tab_num, tab_img, tab_txt = st.tabs([
    "Part A: Numerical (Insurance)",
    "Part B: Images (Scenes)",
    "Part C: Text (BBC News)",
])


# ==========================================
# TAB 1: Numerical Data (Regression)
# ==========================================
with tab_num:
    st.header("Medical Insurance Cost Prediction")

    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", ["female", "male"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    num_model_choice = st.selectbox(
        "Select Regression Model",
        ["Linear Regression", "KNN Regressor"],
    )

    if st.button("Predict Insurance Cost"):
        # Apply the exact same encoding used in numerical/processor.py
        # Note: 'northeast' was dropped during One-Hot Encoding (drop_first=True),
        # so when region is 'northeast', all three region columns are 0.
        input_df = pd.DataFrame([{
            "age": age,
            "sex": 1 if sex == "male" else 0,
            "bmi": bmi,
            "children": children,
            "smoker": 1 if smoker == "yes" else 0,
            "region_northwest": 1 if region == "northwest" else 0,
            "region_southeast": 1 if region == "southeast" else 0,
            "region_southwest": 1 if region == "southwest" else 0,
        }])

        # Scale the features using the saved training scaler
        scaled_input = models["tabular_scaler"].transform(input_df)

        # Pick the selected model and predict
        chosen_model = (
            models["num_lr"] if num_model_choice == "Linear Regression" else models["num_knn"]
        )
        predicted_cost = chosen_model.predict(scaled_input)[0]

        st.success(f"Predicted Insurance Charges: ${predicted_cost:,.2f}")


# ==========================================
# TAB 2: Image Data (Classification)
# ==========================================
with tab_img:
    st.header("Scene Image Classification")
    st.caption("Classes: buildings, forest, mountain, sea, street")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    img_model_choice = st.selectbox(
        "Select Image Classifier",
        ["Logistic Regression", "KNN Classifier"],
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=250)

        if st.button("Classify Image"):
            if img_model_choice == "KNN Classifier" and "img_knn" not in models:
                st.warning(
                    "Image KNN model not found locally. Please run `python images/knn.py` "
                    "on your device first to generate `models/image_knn_model.joblib`."
                )
            else:
                # Read uploaded image bytes into OpenCV and apply the exact same steps
                # used in images/processor.py: BGR->RGB, resize to 64x64, normalize [0, 1]
                file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
                img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img = cv.resize(img, (64, 64), interpolation=cv.INTER_AREA)
                img = img.astype(np.float32) / 255.0

                # Flatten (64, 64, 3) -> (1, 12288) and scale
                img_flat = img.reshape(1, -1)
                img_scaled = models["image_scaler"].transform(img_flat)

                # Predict label index and decode back to class name
                chosen_model = (
                    models["img_lr"] if img_model_choice == "Logistic Regression" else models["img_knn"]
                )
                pred_index = chosen_model.predict(img_scaled)[0]
                pred_label = models["image_le"].inverse_transform([pred_index])[0]

                st.success(f"Predicted Scene: {pred_label.capitalize()}")


# ==========================================
# TAB 3: Text Data (Classification)
# ==========================================
with tab_txt:
    st.header("BBC News Topic Classification")
    st.caption("Classes: business, entertainment, politics, sport, tech")

    user_text = st.text_area("Paste or type a news article:", height=150)

    available_text_models = {"Logistic Regression": models["text_lr"]}
    if "text_svm" in models:
        available_text_models["Linear SVM"] = models["text_svm"]

    txt_model_choice = st.selectbox(
        "Select Text Classifier",
        list(available_text_models.keys()),
    )

    if st.button("Classify Text"):
        if not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            # Clean text using text/preprocessing.py and transform with saved TF-IDF
            cleaned = clean_text(user_text)
            vectorized = models["text_tfidf"].transform([cleaned])

            chosen_model = available_text_models[txt_model_choice]
            pred_topic = chosen_model.predict(vectorized)[0]

            st.success(f"Predicted Topic: {pred_topic.capitalize()}")
