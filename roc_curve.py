import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

X, y = make_blobs(n_samples=1000, centers=2, random_state=42, cluster_std=4)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
criterion = nn.BCELoss()

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    y_scores = model(X_test).numpy().flatten()

y_true = y_test.numpy().flatten()

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc = auc(fpr, tpr)

np.random.seed(42)
random_scores = np.random.rand(len(y_true))
fpr_r, tpr_r, _ = roc_curve(y_true, random_scores)
roc_r = auc(fpr_r, tpr_r)

plt.figure()
plt.plot(fpr, tpr, label=f"NN (AUC = {roc:.2f})")
plt.plot(fpr_r, tpr_r, linestyle='--', label=f"Random (AUC = {roc_r:.2f})")

plt.plot([0, 1], [0, 1], linestyle=':', label="Random Baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve")
plt.legend()
plt.grid()

plt.show()