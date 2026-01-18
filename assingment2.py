"""
KIE4033: AI for Medicine - Assignment 2
Student Project: Intelligent Stress Monitoring System
Target Project: Junior IDP Wearable Health Device
Dataset: Physiological Sensor Data (20,000 samples)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def run_full_pipeline():
    print("--- STARTING AI PIPELINE ---")

    # ===============================
    # STEP 1: DATA LOADING
    # ===============================
    file_path = "strease_synthetic_dataset_20000.csv"

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "CSV file not found. Place it in the same folder as this script."
        )

    df = pd.read_csv(file_path)
    print("Dataset loaded:", df.shape)

    # ===============================
    # STEP 2: PREPROCESSING
    # ===============================
    features = ["ir_mean", "ir_std", "gsr_mean", "gsr_std"]
    X = df[features]
    y = df["stress_label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ===============================
    # STEP 3: EDA
    # ===============================
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.savefig("correlation_analysis.png")
    plt.close()

    # ===============================
    # STEP 4: CLASSIFICATION
    # ===============================
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.savefig("confusion_matrix.png")
        plt.close()
    
        print("--- PIPELINE COMPLETE ---")
    
    
    if __name__ == "__main__":
        run_full_pipeline()
