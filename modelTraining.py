import pandas as pd
from AutoEncoder import AutoEncoder
from trainingconstants import NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn

train_data = pd.read_csv("processedData/X_train_normal.csv")
train_data_tensor = torch.tensor(train_data.values, dtype=torch.float32)
print(train_data.shape)

model = AutoEncoder(train_data.shape[1])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
train_dataset = data.TensorDataset(train_data_tensor, train_data_tensor)
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)

loss_fn = nn.MSELoss()

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    model.train()
    for inputs, _ in train_loader:
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, inputs)
        loss.backward()
        optimizer.step()

        epoch_loss += loss * inputs.size(0)

    epoch_loss = epoch_loss / len(train_loader)
    if epoch % 10 == 0:
        print(f"Epoch: {epoch+1}, Loss: {epoch_loss}")





