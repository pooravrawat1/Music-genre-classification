import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScalar, LabelEncoder
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/processed"
MODEL_PATH = "models/random_forest.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
SCALER_PATH = "models/scaler.pkl"

os.makedirs("models", exist_ok=True)

x_train = pd.read_csv(os.path.join(DATA_PATH, "x_train.csv"))
x_test = pd.read_csv(os.path.join(DATA_PATH, "x_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_PATH, "y_train.csv")).squeeze
y_test = pd.read_csv(os.path.join(DATA_PATH, "y_test.csv")).squeeze()

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Test Accuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figutr(figsize =(10, 7))
sns.heatmap(cm, annot = True, fmt ="d", cmap = "Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

joblib.dump(model, MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")

if ENCODER_PATH.exists() or SCALER_PATH.exists():
    print("Encoder/Scaler already exist, skipping save")
else:
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    joblib.dump(encoder, ENCODER_PATH)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Encoder and Scaler saved for inference at {ENCODER_PATH} and {SCALER_PATH}")


