# 🎓 AI Student Performance Prediction

Predicts whether a student will **Pass** or **Fail**, using and comparing three
ML models: Logistic Regression, Decision Tree, and Random Forest.

## Project Structure
```
student_performance_prediction/
├── data/
│   └── student_data.csv           # dataset (synthetic sample included)
├── notebooks/
│   └── Student_Performance_Prediction.ipynb   # full pipeline, ready to run
├── src/
│   ├── generate_data.py           # creates the synthetic dataset
│   ├── data_preprocessing.py      # cleaning, encoding, scaling, split
│   ├── visualization.py           # EDA plots
│   ├── train_models.py            # trains + compares the 3 models
│   └── predict.py                 # predicts Pass/Fail for a new student
├── models/                        # saved model + encoders + scaler (generated)
├── outputs/figures/                # saved charts (generated)
└── requirements.txt
```

## Features
- Data preprocessing (missing values, encoding, scaling)
- Visualization (EDA: distributions, correlations, class balance)
- Model training (Logistic Regression, Decision Tree, Random Forest)
- Prediction (single-student inference with probability)

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. (Optional) Regenerate the dataset**
A sample synthetic dataset is already in `data/student_data.csv`.
To regenerate it, or to plug in your **own real dataset**, keep the same
column names (see below) and either overwrite that file or point
`src/data_preprocessing.py`'s `DATA_PATH` at your CSV.
```bash
python3 src/generate_data.py
```

**3. Run everything through the notebook (recommended)**
```bash
jupyter notebook notebooks/Student_Performance_Prediction.ipynb
```
This walks through: load data → EDA → preprocess → train & compare models →
evaluate → predict a new student.

**4. Or run the pipeline as scripts**
```bash
cd src
python3 data_preprocessing.py   # test preprocessing
python3 visualization.py        # generate EDA charts
python3 train_models.py         # train, evaluate, save best model
python3 predict.py              # predict on example students
```

## Dataset Columns
| Column | Type | Description |
|---|---|---|
| Hours_Studied | numeric | avg. daily study hours |
| Attendance | numeric | attendance % |
| Previous_Grade | numeric | last exam/term score |
| Sleep_Hours | numeric | avg. sleep hours |
| Parental_Support | categorical | Low / Medium / High |
| Extracurricular | categorical | Yes / No |
| Internet_Access | categorical | Yes / No |
| Study_Group | categorical | Yes / No |
| Result | target | Pass / Fail |

## Model Results (on the included synthetic data)
Logistic Regression currently performs best (~78% accuracy). Exact numbers
will vary depending on the dataset used — re-run `train_models.py` to see
your own comparison table and charts in `outputs/figures/model_comparison.png`.

## Future Improvements
- Flask deployment (`/predict` API endpoint around `src/predict.py`)
- Interactive dashboard (Streamlit/Dash) on top of `outputs/figures`
- Hyperparameter tuning with `GridSearchCV`
- Swap in a real-world dataset (e.g. UCI Student Performance dataset)
