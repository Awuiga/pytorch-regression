import torch
import torch.nn as nn
from sklearn.datasets import load_diabetes

class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )

    def forward(self, features):
        predictions = self.layers(features)
        return predictions

model = RegressionModel()
model_state_dict = torch.load("regression_model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(model_state_dict)
model.eval()

dataset = load_diabetes()
features = torch.tensor(dataset.data[0], dtype = torch.float32)
features = features.unsqueeze(0)
with torch.no_grad():
    prediction = model(features)
print(prediction.item(), dataset.target[0])
