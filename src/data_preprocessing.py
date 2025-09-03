import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

def preprocess():
    df = pd.read_csv('data/processed/features.csv')

    x = df.drop(columns=['label'])
    y = df['label']

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print(list(encoder.classes_))

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    os.makedirs("data/processed", exist_ok=True)
    pd.DataFrame(X_train).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv("data/processed/X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv("data/processed/y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv("data/processed/y_test.csv", index=False)

    print("✅ Preprocessing done! Data saved in:", "data/processed/")

if __name__ == "__main__":
    preprocess()