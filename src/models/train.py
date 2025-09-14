import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Import our models
from src.models.random_forest_model import train_random_forest
from src.models.neural_net_model import train_neural_net
from src.models.knn_model import train_knn

DATA_PATH = "data/processed"

def load_data():
    x = pd.read_csv(os.path.join(DATA_PATH, "x_train.csv"))
    x_test = pd.read_csv(os.path.join(DATA_PATH, "x_test.csv"))
    y = pd.read_csv(os.path.join(DATA_PATH, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(DATA_PATH, "y_test.csv")).squeeze()
    return x, y, x_test, y_test

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data()
    num_classes = len(y_train.unique())

    # Uncomment the one you want to train
    # train_random_forest(x_train, y_train, x_test, y_test)
    # train_knn(x_train, y_train, x_test, y_test)
    train_neural_net(x_train, y_train, x_test, y_test, num_classes)
