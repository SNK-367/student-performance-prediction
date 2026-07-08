"""
train_models.py
------------------
Trains Logistic Regression, Decision Tree, and Random Forest models
on the student performance dataset, evaluates each, and saves the
best-performing model + a comparison chart.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from data_preprocessing import load_and_prepare_data

MODELS_DIR = "/home/claude/student_performance_prediction/models"
FIG_DIR = "/home/claude/student_performance_prediction/outputs/figures"


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    }


def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
    }, preds


def plot_comparison(results_df: pd.DataFrame):
    melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    plt.figure(figsize=(9, 5))
    sns.barplot(data=melted, x="Metric", y="Score", hue="Model", palette="Set2")
    plt.ylim(0, 1)
    plt.title("Model Comparison")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "model_comparison.png"), dpi=120)
    plt.close()


def plot_confusion_matrix(name, y_test, preds):
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(os.path.join(FIG_DIR, f"confusion_matrix_{safe_name}.png"), dpi=120)
    plt.close()


def plot_feature_importance(model, feature_cols):
    if not hasattr(model, "feature_importances_"):
        return
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    plt.figure(figsize=(7, 5))
    sns.barplot(x=importances.values, y=importances.index, hue=importances.index,
                palette="viridis", legend=False)
    plt.title("Feature Importance (Best Model)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "feature_importance.png"), dpi=120)
    plt.close()


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test, scaler, encoders, feature_cols = load_and_prepare_data()

    results = []
    trained_models = {}
    preds_by_model = {}

    for name, model in get_models().items():
        model.fit(X_train, y_train)
        metrics, preds = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)
        trained_models[name] = model
        preds_by_model[name] = preds
        plot_confusion_matrix(name, y_test, preds)
        print(f"\n--- {name} ---")
        print(classification_report(y_test, preds, target_names=["Fail", "Pass"]))

    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    plot_comparison(results_df)

    best_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]

    print("\n=== Model Comparison ===")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_name} (Accuracy: {results_df.iloc[0]['Accuracy']:.3f})")

    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(best_name, os.path.join(MODELS_DIR, "best_model_name.pkl"))
    results_df.to_csv(os.path.join(MODELS_DIR, "model_comparison_results.csv"), index=False)

    plot_feature_importance(best_model, feature_cols)

    print(f"\nSaved best model ({best_name}) and comparison chart to {MODELS_DIR} / {FIG_DIR}")
    return results_df, best_name, best_model


if __name__ == "__main__":
    main()
