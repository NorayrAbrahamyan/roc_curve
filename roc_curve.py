import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

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

def roc_auc(y_true, y_scores):
    thresholds = np.sort(np.unique(y_scores))[::-1]

    tprs, fprs = [], []
    P = y_true.sum() 
    N = len(y_true) - P

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)

        TP = ((y_pred == 1) & (y_true == 1)).sum()
        FP = ((y_pred == 1) & (y_true == 0)).sum()
        FN = ((y_pred == 0) & (y_true == 1)).sum()
        TN = ((y_pred == 0) & (y_true == 0)).sum()

        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0 
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0 
        tprs.append(TPR)
        fprs.append(FPR)

    fprs = np.array([0.0] + fprs + [1.0])
    tprs = np.array([0.0] + tprs + [1.0])

    auc = np.sum(np.diff(fprs) * (tprs[:-1] + tprs[1:]) / 2)
    return fprs, tprs, auc

np.random.seed(42)
random_scores = np.random.rand(len(y_true))

fpr_nn,  tpr_nn,  auc_nn  = roc_auc(y_true, y_scores)
fpr_rnd, tpr_rnd, auc_rnd = roc_auc(y_true, random_scores)

plt.figure()
plt.plot(fpr_nn,  tpr_nn,  label=f"NN (AUC = {auc_nn:.2f})")
plt.plot(fpr_rnd, tpr_rnd, linestyle="--", label=f"Random (AUC = {auc_rnd:.2f})")
plt.plot([0, 1], [0, 1], linestyle=":", label="Random Baseline")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve")
plt.legend()
plt.grid()
plt.show()
