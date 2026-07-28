import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DETECTION_DATA = ROOT / 'datasets' / 'crc_dataset.csv'
STAGE_DATA = ROOT / 'datasets' / 'colorectal_cancer_prediction.csv'
if not STAGE_DATA.exists():
    STAGE_DATA = ROOT / 'colorectal_cancer_prediction.csv'

DETECTION_MODEL_DIR = ROOT / 'saved_models for cancer dectection'
STAGE_MODEL_DIR = ROOT / 'saved_models for stage classification'
DETECTION_MODEL_DIR.mkdir(parents=True, exist_ok=True)
STAGE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


def save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def build_detection_artifacts():
    df = pd.read_csv(DETECTION_DATA)

    def bmi_category(bmi):
        if bmi < 18.5:
            return 'Underweight'
        if bmi < 25:
            return 'Normal'
        if bmi < 30:
            return 'Overweight'
        return 'Obese'

    df_proc = df[['Age', 'BMI', 'Gender', 'Lifestyle', 'Ethnicity', 'Family_History_CRC', 'Pre-existing Conditions', 'Carbohydrates (g)', 'Proteins (g)', 'Fats (g)', 'Vitamin A (IU)', 'Vitamin C (mg)', 'Iron (mg)', 'CRC_Risk']].copy()
    df_proc['BMI_Category'] = df_proc['BMI'].apply(bmi_category)
    df_proc['Age_Group'] = pd.cut(df_proc['Age'], bins=[0, 40, 50, 60, 70, 100], labels=['<40', '40-50', '50-60', '60-70', '70+'])
    df_proc['High_Risk_Lifestyle'] = df_proc['Lifestyle'].isin(['Sedentary', 'Smoker']).astype(int)
    df_proc['Has_Preexisting_Condition'] = (df_proc['Pre-existing Conditions'] != 'None').astype(int)
    df_proc['Nutrition_Risk_Score'] = (
        (df_proc['Fats (g)'] > df_proc['Fats (g)'].median()).astype(int) +
        (df_proc['Vitamin C (mg)'] < df_proc['Vitamin C (mg)'].median()).astype(int) +
        (df_proc['Iron (mg)'] < df_proc['Iron (mg)'].median()).astype(int)
    )

    FEATURES = [
        'Age', 'BMI', 'Carbohydrates (g)', 'Proteins (g)', 'Fats (g)', 'Vitamin A (IU)', 'Vitamin C (mg)', 'Iron (mg)',
        'Gender', 'Lifestyle', 'Ethnicity', 'Family_History_CRC', 'Pre-existing Conditions',
        'BMI_Category', 'Age_Group', 'High_Risk_Lifestyle', 'Has_Preexisting_Condition', 'Nutrition_Risk_Score'
    ]
    TARGET = 'CRC_Risk'

    X = df_proc[FEATURES].copy()
    y = df_proc[TARGET]

    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    le_dict = {}
    for col in cat_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    smote = SMOTE(random_state=SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    model = LGBMClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    save_pickle(DETECTION_MODEL_DIR / 'lightgbm_detection.pkl', model)
    save_pickle(DETECTION_MODEL_DIR / 'scaler_detection.pkl', scaler)
    save_pickle(DETECTION_MODEL_DIR / 'label_encoders_detection.pkl', le_dict)
    save_pickle(DETECTION_MODEL_DIR / 'nb1_features.pkl', FEATURES)
    save_pickle(DETECTION_MODEL_DIR / 'nb1_best_model_name.pkl', 'LightGBM')

    print('Detection model rebuilt')
    print('Accuracy:', report['accuracy'])
    print('F1:', report['1']['f1-score'])
    print('ROC-AUC:', model.predict_proba(X_test)[:, 1][:5].round(4))


def build_stage_artifacts():
    df = pd.read_csv(STAGE_DATA)

    FEATURES = [
        'Age', 'Gender', 'Urban_or_Rural', 'Socioeconomic_Status',
        'Family_History', 'Previous_Cancer_History',
        'Diet_Type', 'BMI', 'Physical_Activity_Level',
        'Smoking_Status', 'Alcohol_Consumption',
        'Red_Meat_Consumption', 'Fiber_Consumption',
    ]
    TARGET = 'Stage_at_Diagnosis'

    df_proc = df[FEATURES + [TARGET]].copy()

    def bmi_cat(b):
        if b < 18.5:
            return 0
        if b < 25:
            return 1
        if b < 30:
            return 2
        return 3

    df_proc['BMI_Category'] = df_proc['BMI'].apply(bmi_cat)
    df_proc['Age_Group'] = pd.cut(df_proc['Age'], bins=[0, 40, 50, 60, 70, 100], labels=[0, 1, 2, 3, 4]).astype(int)
    df_proc['Is_Smoker'] = (df_proc['Smoking_Status'].astype(str).str.lower().str.contains('current')).astype(int)
    df_proc['Is_Sedentary'] = (df_proc['Physical_Activity_Level'].astype(str).str.lower().isin(['low', 'sedentary'])).astype(int)
    df_proc['High_Risk_Diet'] = ((df_proc['Red_Meat_Consumption'] == 'High') & (df_proc['Fiber_Consumption'] == 'Low')).astype(int)
    df_proc['High_Alcohol'] = (df_proc['Alcohol_Consumption'] == 'Heavy').astype(int)
    df_proc['Lifestyle_Risk_Score'] = (
        df_proc['Is_Smoker'] + df_proc['Is_Sedentary'] + df_proc['High_Risk_Diet'] + df_proc['High_Alcohol'] + (df_proc['BMI_Category'] >= 2).astype(int)
    )
    df_proc['Family_or_Prior_Cancer'] = ((df_proc['Family_History'] == 'Yes') | (df_proc['Previous_Cancer_History'] == 'Yes')).astype(int)

    ENGINEERED = ['BMI_Category', 'Age_Group', 'Is_Smoker', 'Is_Sedentary', 'High_Risk_Diet', 'High_Alcohol', 'Lifestyle_Risk_Score', 'Family_or_Prior_Cancer']
    ALL_FEATURES = FEATURES + ENGINEERED

    X = df_proc[ALL_FEATURES].copy()
    y_raw = df_proc[TARGET]
    stage_map = {'I': 0, 'II': 1, 'III': 2, 'IV': 3}
    y = y_raw.map(stage_map)

    cat_cols = X.select_dtypes(include='object').columns.tolist()
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multiclass',
        num_class=4,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, digits=4)

    save_pickle(STAGE_MODEL_DIR / 'lightgbm_stage_classification.pkl', model)
    save_pickle(STAGE_MODEL_DIR / 'scaler_stage.pkl', scaler)
    save_pickle(STAGE_MODEL_DIR / 'label_encoders_stage.pkl', le_dict)
    save_pickle(STAGE_MODEL_DIR / 'nb1_features.pkl', ALL_FEATURES)
    save_pickle(STAGE_MODEL_DIR / 'stage_label_map.pkl', stage_map)

    print('Stage model rebuilt')
    print('Accuracy:', report['accuracy'])
    print('Macro F1:', report['macro avg']['f1-score'])


if __name__ == '__main__':
    build_detection_artifacts()
    build_stage_artifacts()
