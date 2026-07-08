"""
data_preprocessing.py
----------------------
Loads the raw student dataset and prepares it for modeling.
"""

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "student_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ----------------------------
# Columns
# ----------------------------
NUMERIC_COLS = [
    "Hours_Studied",
    "Attendance",
    "Previous_Grade",
    "Sleep_Hours",
]

CATEGORICAL_COLS = [
    "Parental_Support",
    "Extracurricular",
    "Internet_Access",
    "Study_Group",
]

TARGET_COL = "Result"


# ----------------------------
# Load Raw Data
# ----------------------------
def load_raw_data(path=DATA_PATH):

    df = pd.read_csv(path)

    # Rename columns from Kaggle dataset
    df = df.rename(columns={
        "Previous_Scores": "Previous_Grade",
        "Parental_Involvement": "Parental_Support",
        "Extracurricular_Activities": "Extracurricular",
        "Tutoring_Sessions": "Study_Group"
    })

    # Create Pass / Fail target using median score
    threshold = df["Exam_Score"].median()

    df["Result"] = df["Exam_Score"].apply(
        lambda x: "Pass" if x >= threshold else "Fail"
    )

    # Keep only required columns
    df = df[
        [
            "Hours_Studied",
            "Attendance",
            "Previous_Grade",
            "Sleep_Hours",
            "Parental_Support",
            "Extracurricular",
            "Internet_Access",
            "Study_Group",
            "Result",
        ]
    ]

    return df


# ----------------------------
# Clean Data
# ----------------------------
def clean_data(df):

    df = df.copy()

    for col in NUMERIC_COLS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_COLS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


# ----------------------------
# Encode Features
# ----------------------------
def encode_features(df, fit_encoders=True, encoders=None):

    df = df.copy()
    encoders = encoders or {}

    for col in CATEGORICAL_COLS:

        if fit_encoders:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

        else:
            df[col] = encoders[col].transform(df[col].astype(str))

    if TARGET_COL in df.columns:

        if fit_encoders:
            le_target = LabelEncoder()
            df[TARGET_COL] = le_target.fit_transform(df[TARGET_COL])
            encoders[TARGET_COL] = le_target

        else:
            df[TARGET_COL] = encoders[TARGET_COL].transform(df[TARGET_COL])

    return df, encoders


# ----------------------------
# Prepare Train/Test Data
# ----------------------------
def load_and_prepare_data(
    test_size=0.2,
    random_state=42,
    save_artifacts=True,
):

    df = load_raw_data()

    df = clean_data(df)

    df, encoders = encode_features(df)

    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS

    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()

    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index,
    )

    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index,
    )

    if save_artifacts:

        os.makedirs(MODELS_DIR, exist_ok=True)

        joblib.dump(
            scaler,
            os.path.join(MODELS_DIR, "scaler.pkl"),
        )

        joblib.dump(
            encoders,
            os.path.join(MODELS_DIR, "encoders.pkl"),
        )

        joblib.dump(
            feature_cols,
            os.path.join(MODELS_DIR, "feature_columns.pkl"),
        )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        encoders,
        feature_cols,
    )


# ----------------------------
# Test
# ----------------------------
if __name__ == "__main__":

    X_train, X_test, y_train, y_test, scaler, encoders, feature_cols = load_and_prepare_data()

    print("Train Shape:", X_train.shape)
    print("Test Shape :", X_test.shape)
    print("Features   :", feature_cols)
    print("\nClass Distribution")
    print(y_train.value_counts())