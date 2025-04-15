import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

df = pd.read_csv("/kaggle/working/cleaned_customer_data.csv")

threshold = df["MntWines"].quantile(0.75)
df["HighSpender"] = (df["MntWines"] > threshold).astype(int)

df.drop(columns=["ID", "Dt_Customer", "MntWines"], inplace=True)

df = pd.get_dummies(df, drop_first=True)

df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(df.mean(numeric_only=True), inplace=True)

X = df.drop(columns=["HighSpender"])
y = df["HighSpender"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

val_dataset = TensorDataset(X_test_tensor, y_test_tensor)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

input_dim = X_train_tensor.shape[1]
model = LogisticRegressionModel(input_dim)

# Loss and Optimizer (add weight decay for regularization)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

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

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

# Evaluate on test set
model.eval()
with torch.no_grad():
    y_pred_test_logits = model(X_test_tensor)
    y_pred_test_probs = torch.sigmoid(y_pred_test_logits)
    y_pred_test_classes = (y_pred_test_probs > 0.5).float()
    
    acc = accuracy_score(y_test, y_pred_test_classes.numpy())
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred_test_classes.numpy()))
