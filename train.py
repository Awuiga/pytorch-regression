import torch
import torch.nn as nn
from sklearn.datasets import load_diabetes
from torch.utils.data import TensorDataset, random_split, DataLoader

dataset = load_diabetes()
data = dataset.data
target = dataset.target
features = torch.tensor(data, dtype = torch.float32)
targets = torch.tensor(target, dtype = torch.float32)
targets = targets.unsqueeze(1)
dataset = TensorDataset(features, targets)

generator = torch.Generator().manual_seed(42)
train_dataset , val_dataset = random_split(
    dataset,
    [0.8, 0.2],
    generator = generator
)

train_dataloader = DataLoader(train_dataset,
                              batch_size = 16,
                              shuffle = True)

val_dataloader = DataLoader(val_dataset,
                            batch_size = 16,
                            shuffle = False)

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

for features, targets in train_dataloader:
    predictions = model(features)
    break

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001)

model.train()
for epoch in range(100):
    total_loss = 0.0
    for step, (features, targets) in enumerate(train_dataloader):
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_dataloader)

    if (epoch + 1) % 10 == 0:
        print('Epoch', epoch + 1,
            'Loss', loss.item(),
            'Avg_Loss', avg_loss)

model.eval()
val_total_loss = 0.0
with torch.no_grad():
    for features, targets in val_dataloader:
        predictions = model(features)
        loss = criterion(predictions,targets)
        val_total_loss += loss.item()
    avg_val_loss = val_total_loss / len(val_dataloader)
print(avg_val_loss)

torch.save(model.state_dict(), 'regression_model.pt')
