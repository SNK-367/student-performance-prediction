"""
generate_data.py
-----------------
Generates a synthetic but realistic student performance dataset.

Replace this with your real dataset (e.g. a CSV from Kaggle's
"Student Performance" datasets) by simply pointing data_preprocessing.py
at your own file with the same column names, or by editing the
COLUMN MAP at the top of data_preprocessing.py.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 600

hours_studied = np.round(np.random.normal(5, 2.2, N).clip(0, 12), 1)
attendance = np.round(np.random.normal(80, 12, N).clip(40, 100), 1)
previous_grade = np.round(np.random.normal(65, 15, N).clip(30, 100), 1)
sleep_hours = np.round(np.random.normal(6.8, 1.3, N).clip(3, 10), 1)
parental_support = np.random.choice(["Low", "Medium", "High"], N, p=[0.25, 0.45, 0.30])
extracurricular = np.random.choice(["Yes", "No"], N, p=[0.4, 0.6])
internet_access = np.random.choice(["Yes", "No"], N, p=[0.75, 0.25])
study_group = np.random.choice(["Yes", "No"], N, p=[0.3, 0.7])

# Underlying "true" score function + noise, used only to LABEL the data.
support_bonus = {"Low": 0, "Medium": 4, "High": 8}
score = (
    hours_studied * 4.2
    + attendance * 0.35
    + previous_grade * 0.4
    + sleep_hours * 1.5
    + np.array([support_bonus[s] for s in parental_support])
    + np.where(extracurricular == "Yes", 2, 0)
    + np.where(internet_access == "Yes", 3, -2)
    + np.where(study_group == "Yes", 3, 0)
    + np.random.normal(0, 8, N)
)

threshold = np.percentile(score, 35)  # roughly 65% pass rate
result = np.where(score >= threshold, "Pass", "Fail")

df = pd.DataFrame({
    "Hours_Studied": hours_studied,
    "Attendance": attendance,
    "Previous_Grade": previous_grade,
    "Sleep_Hours": sleep_hours,
    "Parental_Support": parental_support,
    "Extracurricular": extracurricular,
    "Internet_Access": internet_access,
    "Study_Group": study_group,
    "Result": result,
})

# Sprinkle a few missing values to make preprocessing realistic
for col in ["Attendance", "Sleep_Hours", "Parental_Support"]:
    idx = np.random.choice(df.index, size=int(0.03 * N), replace=False)
    df.loc[idx, col] = np.nan

out_path = "/home/claude/student_performance_prediction/data/student_data.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(df["Result"].value_counts())
