import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Paths
MODEL_PATH = os.path.join("models", "random_forest.pkl")
DATA_PATH = "data/processed"

def train_random_forest(x_train, y_train, x_test, y_test):
    print("🌲 Training Random Forest...")

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)

    # Predict
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f" Random Forest Accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Random Forest Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f" Random Forest model saved at {MODEL_PATH}")

    return model
