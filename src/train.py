# Import necessary libraries
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier  # Random Forest model
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

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load training and test data from CSV files
x_train = pd.read_csv(os.path.join(DATA_PATH, "x_train.csv"))
x_test = pd.read_csv(os.path.join(DATA_PATH, "x_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_PATH, "y_train.csv")).squeeze() # Squeeze to convert to Series
y_test = pd.read_csv(os.path.join(DATA_PATH, "y_test.csv")).squeeze()

# Initialize and train the Random Forest classifier
# Define parameter grid for tuning
param_grid = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

# Grid search with 5-fold cross-validation
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

# Extract the best model
model = grid_search.best_estimator_
print("✅ Best Parameters:", grid_search.best_params_)
print("✅ Best CV Accuracy:", grid_search.best_score_)

# Predict on test data and evaluate accuracy
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Test Accuracy: {accuracy:.4f}\n")

# Print detailed classification metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Compute and plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize =(10, 7))
sns.heatmap(cm, annot = True, fmt ="d", cmap = "Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save the trained model to disk
joblib.dump(model, MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")

# Save encoder and scaler if they don't already exist
if os.path.exists(ENCODER_PATH) or os.path.exists(SCALER_PATH):
    print("Encoder/Scaler already exist, skipping save")
else:
    encoder = LabelEncoder()  # Initialize label encoder
    y_train_encoded = encoder.fit_transform(y_train)  # Fit and transform labels
    joblib.dump(encoder, ENCODER_PATH)  # Save encoder
    scaler = StandardScaler()  # Initialize scaler
    x_train_scaled = scaler.fit_transform(x_train)  # Fit and transform features
    joblib.dump(scaler, SCALER_PATH)  # Save scaler
    print(f"Encoder and Scaler saved for inference at {ENCODER_PATH} and {SCALER_PATH}")
