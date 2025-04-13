import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset

df = pd.read_csv("/kaggle/working/cleaned_customer_data.csv")

threshold = df["MntWines"].quantile(0.75)
df["HighSpender"] = (df["MntWines"] > threshold).astype(int)
df.drop(columns=["ID", "Dt_Customer", "MntWines"], inplace=True)

X = df.drop("HighSpender", axis=1)
y = df["HighSpender"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.astype({col: 'int' for col in X_train.select_dtypes(include='bool').columns})
X_test = X_test.astype({col: 'int' for col in X_test.select_dtypes(include='bool').columns})

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

df = pd.get_dummies(df, drop_first=True)
df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(df.mean(), inplace=True)

X = torch.tensor(df.drop(columns=['Response']).values, dtype=torch.float32)
y = torch.tensor(df['Response'].values, dtype=torch.float32).view(-1, 1)

dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

model = LogisticRegressionModel(input_dim=X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

epochs = 50
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

with torch.no_grad():
    model.eval()
    y_pred = model(X)
    y_pred_class = (y_pred > 0.5).float()

    accuracy = (y_pred_class == y).float().mean()
    print(f'Accuracy: {accuracy.item():.4f}') # 85.09


with torch.no_grad():
    model.eval()
    y_pred_test = model(torch.tensor(X_test.values, dtype=torch.float32))  # Use X_test
    y_pred_class_test = (y_pred_test > 0.5).float()

    accuracy_test = (y_pred_class_test == torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)).float().mean()
    print(f'Test Accuracy: {accuracy_test.item():.4f}') # 70.04
