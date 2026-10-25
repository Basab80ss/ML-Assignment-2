import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import os

# 1. Seed extraction from Roll Number: m26cse002 -> 26002
ROLL_NO = "m26cse002"
SEED = int("".join([c for c in ROLL_NO if c.isdigit()]))
print(f"Roll Number: {ROLL_NO}")
print(f"Extracted Seed: {SEED}")

# 2. Load dataset
faces = fetch_olivetti_faces(shuffle=False)
X_all = faces.data # 400 x 4096 (raw flattened pixels)
y_all = faces.target # 400, 40 identities (0 to 39), 10 each

rng = np.random.default_rng(SEED)

# 3. Stratified split within each identity: 6 train, 2 val, 2 test
train_idx = []
val_idx = []
test_idx = []

for identity in range(40):
    identity_indices = np.where(y_all == identity)[0]
    shuffled = rng.permutation(identity_indices)
    train_idx.extend(shuffled[:6])
    val_idx.extend(shuffled[6:8])
    test_idx.extend(shuffled[8:])

train_idx = np.array(train_idx)
val_idx = np.array(val_idx)
test_idx = np.array(test_idx)

print(f"Train samples: {len(train_idx)} (6 per identity)")
print(f"Val samples:   {len(val_idx)} (2 per identity)")
print(f"Test samples:  {len(test_idx)} (2 per identity)")

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_val, y_val = X_all[val_idx], y_all[val_idx]
X_test, y_test = X_all[test_idx], y_all[test_idx]

# Create masked test set: centered 16x16 block (rows 24 to 39, cols 24 to 39) set to 0
X_test_masked = X_test.copy()
for i in range(len(X_test_masked)):
    img = X_test_masked[i].reshape(64, 64)
    img[24:40, 24:40] = 0.0
    X_test_masked[i] = img.flatten()

# Save the split so it can be reloaded everywhere
np.savez("split_data.npz", 
         train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
         X_train=X_train, y_train=y_train,
         X_val=X_val, y_val=y_val,
         X_test=X_test, y_test=y_test,
         X_test_masked=X_test_masked)
print("Saved split_data.npz successfully.")
