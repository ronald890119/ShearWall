import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torcheval.metrics import R2Score
import matplotlib.pyplot as plt

# load dataset
df= pd.read_csv('concrete.csv')

# remove outliers/noises
df=df[df['Blast Furnace Slag'] < 350]
df=df[(df['Water'] > 125) & (df['Water'] < 230)]
df=df[df['Superplasticizer'] < 25]
df=df[df['Fine Aggregate'] < 960]
df=df[df['Age'] < 150]

X = df.drop('Strength', axis=1)
Y = df['Strength']

standard = StandardScaler()

# split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2024)
standard = standard.fit(X_train)
X_train = torch.tensor(standard.fit_transform(X_train), dtype=torch.float32)
X_test = torch.tensor(standard.fit_transform(X_test), dtype=torch.float32)
Y_train = torch.tensor(Y_train.to_numpy(), dtype=torch.float32).reshape(-1, 1)
Y_test = torch.tensor(Y_test.to_numpy(), dtype=torch.float32).reshape(-1, 1)

model = nn.Sequential(
    nn.Linear(8, 32),
    nn.LeakyReLU(),
    nn.Linear(32, 64),
    nn.LeakyReLU(),
    nn.Linear(64, 128),
    nn.LeakyReLU(),
    nn.Linear(128, 32),
    nn.LeakyReLU(),
    nn.Linear(32, 16),
    nn.LeakyReLU(),
    nn.Linear(16, 1),
)

# MSE as loss function
loss_func = nn.MSELoss()
# R-squared for validation
metric = R2Score()
optimiser = optim.AdamW(model.parameters(), lr=0.001)
epoch = 1500
batch = 15
batch_start = torch.arange(0, len(X_train), batch)
avg_train_accuracy = []
test_accuracy = []

for epoch in range(epoch):
    # set model to training mode
    model.train()
    total_r2 = 0
    for start in batch_start:
        # batches for training
        X_batch = X_train[start:start+batch]
        Y_batch = Y_train[start:start+batch]
        # predict and calculate the loss
        Y_pred = model(X_batch)
        loss = loss_func(Y_pred, Y_batch)
        metric.update(Y_pred, Y_batch)
        total_r2 += float(metric.compute())
        # backward propagation
        optimiser.zero_grad()
        loss.backward()
        # update weights
        optimiser.step()
    
    avg_train_accuracy.append(total_r2 / len(batch_start))
    
    # set model to evaluation mode
    model.eval()
    Y_pred = model(X_test)
    metric.update(Y_pred, Y_test)
    r2_val = float(metric.compute())
    test_accuracy.append(r2_val)
    print(f'Epoch #{epoch} with validation r2 score: {r2_val}')

# plot the R-squared score for training and validation
plt.plot(avg_train_accuracy, label = 'train_r2')
plt.plot(test_accuracy, label = 'val_r2')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('R-squared score')
plt.show()

# Epoch #999 with validation r2 score around 0.93 ~ 0.94
# Epoch 1500 with validation r2 score around 0.95206

# save the parameters of model and optimiser
torch.save(model.state_dict(), f'model#{epoch}_{round(r2_val * 100)}.pth')
torch.save(optimiser.state_dict(), f'optimiser#{epoch}_{round(r2_val * 100)}.pth')