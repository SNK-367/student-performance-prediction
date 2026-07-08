"""
visualization.py
------------------
Exploratory data analysis plots for the student performance dataset.
Saves figures to outputs/figures/ so they can be reused in reports
or the notebook without re-running training.
"""

import matplotlib
matplotlib.use("Agg")  # safe for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from data_preprocessing import load_raw_data, clean_data, NUMERIC_COLS, CATEGORICAL_COLS, TARGET_COL

FIG_DIR = "/home/claude/student_performance_prediction/outputs/figures"
sns.set_style("whitegrid")


def plot_target_balance(df: pd.DataFrame):
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x=TARGET_COL, hue=TARGET_COL, palette="Set2", legend=False)
    plt.title("Pass vs Fail Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "target_balance.png"), dpi=120)
    plt.close()


def plot_numeric_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.flat, NUMERIC_COLS):
        sns.histplot(data=df, x=col, hue=TARGET_COL, kde=True, ax=ax, palette="Set2", multiple="stack")
        ax.set_title(f"{col} by Result")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "numeric_distributions.png"), dpi=120)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_df = df[NUMERIC_COLS].copy()
    plt.figure(figsize=(6, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Between Numeric Features")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "correlation_heatmap.png"), dpi=120)
    plt.close()


def plot_categorical_vs_result(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.flat, CATEGORICAL_COLS):
        sns.countplot(data=df, x=col, hue=TARGET_COL, ax=ax, palette="Set2")
        ax.set_title(f"{col} vs Result")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "categorical_vs_result.png"), dpi=120)
    plt.close()


def run_all_eda():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = clean_data(load_raw_data())
    plot_target_balance(df)
    plot_numeric_distributions(df)
    plot_correlation_heatmap(df)
    plot_categorical_vs_result(df)
    print(f"Saved 4 EDA figures to {FIG_DIR}")


if __name__ == "__main__":
    run_all_eda()
