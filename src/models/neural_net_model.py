import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODEL_PATH = os.path.join("models", "neural_net.h5")

def train_neural_net(x_train, y_train, x_test, y_test, num_classes):
    print("🧠 Training Neural Network...")

    # One-hot encode labels
    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)

    model = keras.Sequential([
        keras.layers.Dense(256, activation="relu", input_shape=(x_train.shape[1],)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(
        x_train, y_train_cat,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # Evaluate
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Neural Net Accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Neural Net Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Save model
    model.save(MODEL_PATH)
    print(f"💾 Neural Net model saved at {MODEL_PATH}")

    return model, history
