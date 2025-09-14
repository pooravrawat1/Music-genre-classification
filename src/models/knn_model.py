import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_PATH = os.path.join("models", "knn.pkl")

def train_knn(x_train, y_train, x_test, y_test):
    print("👥 Training KNN...")

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train, y_train)

    # Predict
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ KNN Accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("KNN Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"💾 KNN model saved at {MODEL_PATH}")

    return model
