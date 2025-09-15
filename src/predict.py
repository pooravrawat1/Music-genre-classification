import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse

# 1. Define the model (must match the architecture used in training)
class SimpleNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, 784)  # flatten (28x28 -> 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. Preprocessing transform (same as training)
transform = transforms.Compose([
    transforms.Grayscale(),           # make sure it's single channel
    transforms.Resize((28, 28)),      # resize to 28x28
    transforms.ToTensor(),            # convert to tensor
    transforms.Normalize((0.5,), (0.5,)) # normalize
])

def predict(image_path, model_path="models/random_forest.pkl"):
    # Load the model
    model = SimpleNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # set to eval mode

    # Load and preprocess the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # add batch dimension

    # Run inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using trained model")
    parser.add_argument("image", type=str, help="Path to input image")
    args = parser.parse_args()

    prediction = predict(args.image)
    print(f"Predicted class: {prediction}")
