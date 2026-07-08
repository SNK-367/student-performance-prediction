"""
predict.py
------------
Loads the saved model + preprocessing artifacts and predicts Pass/Fail
for one or more new students.

Usage (as a script):
    python3 predict.py

Usage (importable function):
    from predict import predict_student
    predict_student({
        "Hours_Studied": 6.5,
        "Attendance": 88,
        "Previous_Grade": 72,
        "Sleep_Hours": 7,
        "Parental_Support": "High",
        "Extracurricular": "Yes",
        "Internet_Access": "Yes",
        "Study_Group": "No",
    })
"""

import pandas as pd
import joblib
import os

MODELS_DIR = "/home/claude/student_performance_prediction/models"


def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    best_model_name = joblib.load(os.path.join(MODELS_DIR, "best_model_name.pkl"))
    return model, scaler, encoders, feature_cols, best_model_name


def predict_student(student: dict) -> dict:
    """
    student: dict with keys matching the raw feature columns, e.g.
        Hours_Studied, Attendance, Previous_Grade, Sleep_Hours,
        Parental_Support, Extracurricular, Internet_Access, Study_Group
    Returns dict with prediction label and pass probability.
    """
    model, scaler, encoders, feature_cols, best_model_name = load_artifacts()

    df = pd.DataFrame([student])

    categorical_cols = ["Parental_Support", "Extracurricular", "Internet_Access", "Study_Group"]
    for col in categorical_cols:
        df[col] = encoders[col].transform(df[col])

    df = df[feature_cols]
    df_scaled = pd.DataFrame(scaler.transform(df), columns=feature_cols)

    pred = model.predict(df_scaled)[0]
    proba = model.predict_proba(df_scaled)[0]

    result_label = encoders["Result"].inverse_transform([pred])[0]
    pass_idx = list(encoders["Result"].classes_).index("Pass")

    return {
        "model_used": best_model_name,
        "prediction": result_label,
        "pass_probability": round(float(proba[pass_idx]), 3),
    }


if __name__ == "__main__":
    example_students = [
        {
            "Hours_Studied": 8.0, "Attendance": 92, "Previous_Grade": 78,
            "Sleep_Hours": 7.5, "Parental_Support": "High",
            "Extracurricular": "Yes", "Internet_Access": "Yes", "Study_Group": "Yes",
        },
        {
            "Hours_Studied": 2.0, "Attendance": 55, "Previous_Grade": 40,
            "Sleep_Hours": 5.0, "Parental_Support": "Low",
            "Extracurricular": "No", "Internet_Access": "No", "Study_Group": "No",
        },
    ]

    for i, student in enumerate(example_students, 1):
        result = predict_student(student)
        print(f"Student {i}: {result}")
