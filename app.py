from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import pickle
import numpy as np
import pandas as pd
import shap
from analysis import (
    load_data,
    compute_key_stats,
    plot_km_group_base64,
    plot_cox_forest_base64,
    csv_to_html_table,
)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "devkey")

DATA_PATH = os.path.join("datasets", "colorectal_cancer_prediction.csv")
PLOTS_DIR = os.path.join("static", "plots")
DETECTION_MODEL_DIR = os.path.join("saved_models for cancer dectection")
STAGE_MODEL_DIR = os.path.join("saved_models for stage classification")
SURVIVAL_DATA_DIR = os.path.join("saved_models for survival analysis")
SURVIVAL_SUBGROUP_CSV = os.path.join(SURVIVAL_DATA_DIR, "nb3_subgroup_summary.csv")
SURVIVAL_TREATMENT_CSV = os.path.join(SURVIVAL_DATA_DIR, "nb3_treatment_combos.csv")
SURVIVAL_LOGLRANK_CSV = os.path.join(SURVIVAL_DATA_DIR, "nb3_logrank_results.csv")


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_detection_model():
    try:
        model = load_pickle(os.path.join(DETECTION_MODEL_DIR, "lightgbm_detection.pkl"))
        scaler = load_pickle(os.path.join(DETECTION_MODEL_DIR, "scaler_detection.pkl"))
        le_dict = load_pickle(os.path.join(DETECTION_MODEL_DIR, "label_encoders_detection.pkl"))
        features = load_pickle(os.path.join(DETECTION_MODEL_DIR, "nb1_features.pkl"))
        return model, scaler, le_dict, features
    except Exception as exc:
        print(f"Error loading detection model: {exc}")
        return None, None, None, None


def load_stage_model():
    try:
        model = load_pickle(os.path.join(STAGE_MODEL_DIR, "lightgbm_stage_classification.pkl"))
        scaler = load_pickle(os.path.join(STAGE_MODEL_DIR, "scaler_stage.pkl"))
        le_dict = load_pickle(os.path.join(STAGE_MODEL_DIR, "label_encoders_stage.pkl"))
        stage_map = load_pickle(os.path.join(STAGE_MODEL_DIR, "stage_label_map.pkl"))
        features_path = os.path.join(STAGE_MODEL_DIR, "nb1_features.pkl")
        features = load_pickle(features_path) if os.path.exists(features_path) else None
        return model, scaler, le_dict, stage_map, features
    except Exception as exc:
        print(f"Error loading stage model: {exc}")
        return None, None, None, None, None


DETECTION_MODEL, DETECTION_SCALER, DETECTION_LE, DETECTION_FEATURES = load_detection_model()
STAGE_MODEL, STAGE_SCALER, STAGE_LE, STAGE_MAP, STAGE_FEATURES = load_stage_model()

try:
    detection_training_df = pd.read_csv(os.path.join("datasets", "crc_dataset.csv"))
    DETECTION_MEDIANS = {
        "Fats (g)": float(detection_training_df["Fats (g)"].median()),
        "Vitamin C (mg)": float(detection_training_df["Vitamin C (mg)"].median()),
        "Iron (mg)": float(detection_training_df["Iron (mg)"].median()),
    }
except Exception as exc:
    print(f"Could not load detection medians: {exc}")
    DETECTION_MEDIANS = {"Fats (g)": 70.0, "Vitamin C (mg)": 90.0, "Iron (mg)": 18.0}


def build_detection_frame(data):
    age = float(data.get("age", 60))
    bmi = float(data.get("bmi", 26))
    gender = str(data.get("gender", "Male")).strip() or "Male"
    smoke = int(data.get("smoke", 0))
    family = int(data.get("family", 0))
    preexist = int(data.get("preexist", 0))
    carbs = float(data.get("carbs", 250))
    protein = float(data.get("protein", 80))
    fat = float(data.get("fat", 70))
    vita = float(data.get("vitamin_a", 800))
    vitc = float(data.get("vitamin_c", 90))
    iron = float(data.get("iron", 18))
    ethnicity = str(data.get("ethnicity", "Caucasian")).strip() or "Caucasian"

    if gender not in {"Male", "Female"}:
        gender = "Female"

    lifestyle = "Smoker" if smoke == 2 else ("Sedentary" if smoke == 0 else "Moderate Exercise")
    family_history = "Yes" if family else "No"
    preexisting = "None"
    if preexist == 1:
        preexisting = "Hypertension"
    elif preexist == 2:
        preexisting = "Diabetes"

    if bmi < 18.5:
        bmi_category = "Underweight"
    elif bmi < 25:
        bmi_category = "Normal"
    elif bmi < 30:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"

    if age < 40:
        age_group = "<40"
    elif age < 50:
        age_group = "40-50"
    elif age < 60:
        age_group = "50-60"
    elif age < 70:
        age_group = "60-70"
    else:
        age_group = "70+"

    feature_dict = {
        "Age": age,
        "BMI": bmi,
        "Gender": gender,
        "Carbohydrates (g)": carbs,
        "Proteins (g)": protein,
        "Fats (g)": fat,
        "Vitamin A (IU)": vita,
        "Vitamin C (mg)": vitc,
        "Iron (mg)": iron,
        "Lifestyle": lifestyle,
        "Ethnicity": ethnicity,
        "Family_History_CRC": family_history,
        "Pre-existing Conditions": preexisting,
        "BMI_Category": bmi_category,
        "Age_Group": age_group,
        "High_Risk_Lifestyle": 1 if lifestyle in {"Sedentary", "Smoker"} else 0,
        "Has_Preexisting_Condition": 1 if preexisting != "None" else 0,
    }
    feature_dict["Nutrition_Risk_Score"] = (
        int(fat > DETECTION_MEDIANS["Fats (g)"])
        + int(vitc < DETECTION_MEDIANS["Vitamin C (mg)"])
        + int(iron < DETECTION_MEDIANS["Iron (mg)"])
    )

    frame = pd.DataFrame([feature_dict])
    for col, encoder in (DETECTION_LE or {}).items():
        if col in frame.columns:
            frame[col] = encoder.transform(frame[col].astype(str))
    return frame


def build_stage_frame(data):
    age = float(data.get("age", 60))
    bmi = float(data.get("bmi", 26))

    smoking_value = data.get("smoke", data.get("smoking_status", "Never Smoked"))
    if isinstance(smoking_value, str):
        smoking_text = smoking_value.strip().lower()
        if "current" in smoking_text:
            smoking_status = "Current"
        elif "former" in smoking_text:
            smoking_status = "Former"
        else:
            smoking_status = "Never"
    else:
        smoking_status = {0: "Never", 1: "Former", 2: "Current"}.get(int(smoking_value), "Never")

    diet_type = data.get("diet_type", data.get("diet", "Mediterranean"))
    diet_text = str(diet_type).strip().lower()
    if diet_text in {"high fiber", "vegetarian", "mediterranean", "balanced"}:
        diet_value = "Balanced"
    elif diet_text == "western":
        diet_value = "Western"
    else:
        diet_value = "Traditional"

    activity_value = data.get("physical_activity", data.get("activity", "Moderately Active"))
    activity_text = str(activity_value).strip().lower()
    if activity_text in {"sedentary", "lightly active", "low"}:
        physical_activity = "Low"
    elif activity_text in {"moderately active", "medium"}:
        physical_activity = "Medium"
    else:
        physical_activity = "High"

    ses_value = data.get("ses", data.get("socioeconomic_status", "Middle"))
    ses_text = str(ses_value).strip().lower()
    if ses_text in {"low", "l"}:
        socioeconomic_status = "Low"
    elif ses_text in {"high", "h"}:
        socioeconomic_status = "High"
    else:
        socioeconomic_status = "Middle"

    gender = str(data.get("gender", "Male")).strip() or "Male"
    if gender not in {"Male", "Female"}:
        gender = "Female"

    urban_or_rural = str(data.get("urban_or_rural", "Urban")).strip() or "Urban"
    family_history = "Yes" if str(data.get("family_history", "No")).strip().lower() in {"1", "yes", "true"} else "No"
    previous_cancer_history = "Yes" if str(data.get("previous_cancer_history", "No")).strip().lower() in {"1", "yes", "true"} else "No"

    alcohol_value = data.get("alcohol", data.get("Alcohol_Consumption", "None"))
    alcohol_text = str(alcohol_value).strip().lower()
    if alcohol_text in {"heavy", "high", "2"}:
        alcohol_consumption = "High"
    elif alcohol_text in {"moderate", "medium", "1"}:
        alcohol_consumption = "Medium"
    else:
        alcohol_consumption = "Low"

    red_meat = str(data.get("red_meat", data.get("Red_Meat_Consumption", "Medium"))).strip() or "Medium"
    fiber = str(data.get("fiber", data.get("Fiber_Consumption", "Medium"))).strip() or "Medium"

    if bmi < 18.5:
        bmi_category = 0
    elif bmi < 25:
        bmi_category = 1
    elif bmi < 30:
        bmi_category = 2
    else:
        bmi_category = 3

    if age < 40:
        age_group = 0
    elif age < 50:
        age_group = 1
    elif age < 60:
        age_group = 2
    elif age < 70:
        age_group = 3
    else:
        age_group = 4

    is_smoker = int(smoking_status == "Current")
    is_sedentary = int(physical_activity == "Low")
    high_risk_diet = int((red_meat == "High") and (fiber == "Low"))
    high_alcohol = int(alcohol_consumption == "High")
    lifestyle_risk_score = is_smoker + is_sedentary + high_risk_diet + high_alcohol + int(bmi_category >= 2)
    family_or_prior_cancer = int(family_history == "Yes" or previous_cancer_history == "Yes")

    feature_dict = {
        "Age": age,
        "Gender": gender,
        "Urban_or_Rural": urban_or_rural,
        "Socioeconomic_Status": socioeconomic_status,
        "Family_History": family_history,
        "Previous_Cancer_History": previous_cancer_history,
        "Diet_Type": diet_value,
        "BMI": bmi,
        "Physical_Activity_Level": physical_activity,
        "Smoking_Status": smoking_status,
        "Alcohol_Consumption": alcohol_consumption,
        "Red_Meat_Consumption": red_meat,
        "Fiber_Consumption": fiber,
        "BMI_Category": bmi_category,
        "Age_Group": age_group,
        "Is_Smoker": is_smoker,
        "Is_Sedentary": is_sedentary,
        "High_Risk_Diet": high_risk_diet,
        "High_Alcohol": high_alcohol,
        "Lifestyle_Risk_Score": lifestyle_risk_score,
        "Family_or_Prior_Cancer": family_or_prior_cancer,
    }
    frame = pd.DataFrame([feature_dict])
    for col, encoder in (STAGE_LE or {}).items():
        if col in frame.columns:
            frame[col] = encoder.transform(frame[col].astype(str))
    return frame


@app.route("/")
def home():
    return redirect(url_for("detection"))


@app.route("/detection", methods=["GET", "POST"])
def detection():
    message = None
    model_ok = DETECTION_MODEL is not None
    if not model_ok:
        message = "⚠ Detection model not loaded. Check saved_models directory."
    return render_template("detection.html", message=message, model_ok=model_ok)


@app.route("/stage", methods=["GET", "POST"])
def stage():
    message = None
    model_ok = STAGE_MODEL is not None
    if not model_ok:
        message = "⚠ Stage model not loaded. Check saved_models directory."
    return render_template("stage_new.html", message=message, model_ok=model_ok)


@app.route("/api/predict-risk", methods=["POST"])
def predict_risk():
    if DETECTION_MODEL is None:
        return jsonify({"error": "Model not loaded"}), 400
    try:
        payload = request.get_json(silent=True) or {}
        X_df = build_detection_frame(payload)
        X_input = X_df[DETECTION_FEATURES].fillna(0)

        prob = DETECTION_MODEL.predict_proba(X_input)[0][1]
        risk_label = "High Risk" if prob >= 0.5 else "Low Risk"

        importances = {}
        try:
            explainer = shap.TreeExplainer(DETECTION_MODEL)
            shap_vals = explainer.shap_values(X_input)
            if isinstance(shap_vals, list) and len(shap_vals) > 1:
                shap_vals = shap_vals[1]
            shap_vals = np.array(shap_vals)
            if shap_vals.ndim == 2:
                shap_vals = shap_vals[0]
            for idx, feat in enumerate(DETECTION_FEATURES[:5]):
                importances[feat] = float(shap_vals[idx])
        except Exception:
            feature_importances = getattr(DETECTION_MODEL, "feature_importances_", None)
            if feature_importances is not None:
                for idx, feat in enumerate(DETECTION_FEATURES[:5]):
                    importances[feat] = float(feature_importances[idx])
            else:
                for idx, feat in enumerate(DETECTION_FEATURES[:5]):
                    importances[feat] = float(idx + 1)

        return jsonify({"risk": risk_label, "probability": float(prob), "importance": importances})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


def build_stage_explanation(stage, payload):
    reasons = []
    smoke = str(payload.get("smoke", "")).strip().lower()
    activity = str(payload.get("physical_activity", payload.get("activity", ""))).strip().lower()
    diet = str(payload.get("diet_type", payload.get("diet", ""))).strip().lower()
    red_meat = str(payload.get("red_meat", payload.get("Red_Meat_Consumption", ""))).strip().lower()
    fiber = str(payload.get("fiber", payload.get("Fiber_Consumption", ""))).strip().lower()
    family_history = str(payload.get("family_history", "no")).strip().lower()
    prior_cancer = str(payload.get("previous_cancer_history", "no")).strip().lower()
    ses = str(payload.get("ses", payload.get("socioeconomic_status", ""))).strip().lower()
    bmi = float(payload.get("bmi", 0) or 0)

    if smoke in {"current smoker", "current", "smoker"}:
        reasons.append("Current smoking is a strong risk factor for more advanced CRC stages.")
    elif smoke in {"former smoker", "former"}:
        reasons.append("Former smoking history still contributes to elevated CRC risk.")

    if activity in {"sedentary", "low", "lightly active"}:
        reasons.append("Low physical activity is associated with poorer colorectal cancer outcomes.")

    if diet in {"western", "high fat", "low fiber"}:
        reasons.append("A less healthy diet pattern may increase progression risk.")

    if red_meat == "high":
        reasons.append("High red meat consumption can raise the likelihood of advanced disease.")
    if fiber == "low":
        reasons.append("Low fiber intake is linked to worse colorectal cancer prognosis.")

    if bmi >= 30:
        reasons.append("Obesity is an important risk factor for colorectal cancer progression.")
    elif bmi >= 25:
        reasons.append("Overweight status can contribute to an elevated colorectal cancer stage.")

    if family_history in {"yes", "1", "true"}:
        reasons.append("A family history of CRC increases the chances of more serious disease.")
    if prior_cancer in {"yes", "1", "true"}:
        reasons.append("Previous cancer history may indicate a higher risk of advanced staging.")

    if ses == "low":
        reasons.append("Lower socioeconomic status may reduce early screening and raise stage risk.")

    if not reasons:
        reasons.append("The prediction is based on the patient’s overall clinical and lifestyle profile.")

    stage_summary = {
        "I": "Stage I suggests early localized disease with the best treatment outlook.",
        "II": "Stage II suggests locally advanced disease without extensive regional spread.",
        "III": "Stage III suggests likely regional lymph node involvement and more aggressive disease.",
        "IV": "Stage IV suggests metastatic disease, which requires urgent multidisciplinary care.",
    }

    urgency = ""
    if stage == "III":
        urgency = "Urgent specialist referral recommended."
    elif stage == "IV":
        urgency = "Immediate oncologist referral and advanced care planning recommended."
    elif stage == "II":
        urgency = "Specialist referral advised to confirm extent and plan therapy."
    else:
        urgency = "Early-stage monitoring and specialist consultation are recommended."

    explanation = f"{stage_summary.get(stage, '')} {urgency} "
    explanation += "Reasons include: " + " ".join(reasons)
    return explanation, urgency


@app.route("/api/predict-stage", methods=["POST"])
def predict_stage():
    if STAGE_MODEL is None:
        return jsonify({"error": "Model not loaded"}), 400
    try:
        payload = request.get_json(silent=True) or {}
        X_df = build_stage_frame(payload)
        X_input = X_df[STAGE_FEATURES].fillna(0)
        probs = STAGE_MODEL.predict_proba(X_input)[0]
        pred = int(STAGE_MODEL.predict(X_input)[0])

        stage_labels = ["I", "II", "III", "IV"]
        stage = stage_labels[min(pred, len(stage_labels) - 1)]
        explanation, urgency = build_stage_explanation(stage, payload)
        return jsonify({
            "stage": stage,
            "probabilities": {stage_labels[idx]: float(probs[idx]) for idx in range(len(probs))},
            "urgency": urgency,
            "explanation": explanation,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/survival")
def survival_dashboard():
    df = load_data(DATA_PATH)
    stats = compute_key_stats(df)

    km_groups = [
        "Stage_at_Diagnosis",
        "Treatment_Access",
        "Socioeconomic_Status",
        "Follow_Up_Adherence",
        "Insurance_Coverage",
        "Smoking_Status",
    ]

    km_group_images = {}
    for group in km_groups:
        km_group_images[group] = plot_km_group_base64(df, group)

    cox_image = plot_cox_forest_base64(df)

    treatment_table = csv_to_html_table(SURVIVAL_TREATMENT_CSV)
    subgroup_table = csv_to_html_table(SURVIVAL_SUBGROUP_CSV)
    logrank_table = csv_to_html_table(SURVIVAL_LOGLRANK_CSV)

    return render_template(
        "survival.html",
        stats=stats,
        km_groups=km_groups,
        km_group_images=km_group_images,
        cox_image=cox_image,
        treatment_table=treatment_table,
        subgroup_table=subgroup_table,
        logrank_table=logrank_table,
    )


if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)
    app.run(debug=False, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
