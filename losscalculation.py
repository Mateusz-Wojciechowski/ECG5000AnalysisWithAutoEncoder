import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from trainingconstants import BATCH_SIZE
from AutoEncoder import AutoEncoder
import torch.nn as nn
import matplotlib.pyplot as plt

model = AutoEncoder(140)
model.load_state_dict(torch.load('model.pth'))
model.eval()

loss_fn = nn.MSELoss(reduction='none')

normal_test_data = pd.read_csv("processedDataModified/X_normal_test.csv")
anomaly_test_data = pd.read_csv("processedDataModified/X_anomalies_test.csv")
normal_test_tensor = torch.tensor(normal_test_data.values, dtype=torch.float32)
anomaly_test_tensor = torch.tensor(anomaly_test_data.values, dtype=torch.float32)

normal_test_loader = DataLoader(TensorDataset(normal_test_tensor, normal_test_tensor), batch_size=BATCH_SIZE, shuffle=False)
anomaly_test_loader = DataLoader(TensorDataset(anomaly_test_tensor, anomaly_test_tensor), batch_size=BATCH_SIZE, shuffle=False)

normal_test_losses = []
with torch.no_grad():
    for inputs, _ in normal_test_loader:
        outputs = model(inputs)
        losses = loss_fn(outputs, inputs)
        normal_test_losses.extend(losses.sum(dim=1).detach().cpu().numpy())


anomaly_test_losses = []
with torch.no_grad():
    for inputs, _ in anomaly_test_loader:
        outputs = model(inputs)
        losses = loss_fn(outputs, inputs)
        anomaly_test_losses.extend(losses.sum(dim=1).detach().cpu().numpy())


plt.figure(figsize=(10, 6))

plt.hist(normal_test_losses, bins=50, alpha=0.5, color='blue', label='Normal')

plt.hist(anomaly_test_losses, bins=50, alpha=0.5, color='red', label='Anomaly')

plt.xlabel('Loss')
plt.ylabel('Frequency')
plt.title('Distribution of Losses')
plt.legend()
plt.grid(True)
plt.show()