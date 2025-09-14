import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier  # Random Forest model
from sklearn.neighbors import KNeighborsClassifier  # KNN 
from xgboost import XGBClassifier # XGBoost model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Evaluation metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Preprocessing tools
import joblib  # For saving/loading models
import os  # OS operations
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns  # Visualization

# Define paths for data and models
DATA_PATH = "data/processed"
MODEL_PATH = "models/random_forest.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
SCALER_PATH = "models/scaler.pkl"
XGB_MODEL_PATH = "models/xgboost.pkl"


# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

def train_random_forest(x_train, y_train, x_test, y_test):
    """
    Train and evaluate a Random Forest classifier with hyperparameter tuning.
    """
    param_grid = {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        verbose=2,
        n_jobs=-1
    )

    print("🔍 Running hyperparameter tuning...")
    grid_search.fit(x_train, y_train)

    model = grid_search.best_estimator_
    print("✅ Best Parameters:", grid_search.best_params_)
    print("✅ Best CV Accuracy:", grid_search.best_score_)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Test Accuracy: {accuracy:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")

    return model