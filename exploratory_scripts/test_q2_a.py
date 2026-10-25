import numpy as np
from sklearn.tree import DecisionTreeClassifier

print("=== QUESTION 2(a): Toy 3-Class Example with at most 10 samples ===")

# Create a small 3-class toy dataset (N = 9 samples, 3 per class, 2 features)
X_toy = np.array([
    [1.0, 2.0], [1.5, 1.8], [1.2, 2.5],  # Class 0
    [5.0, 8.0], [5.5, 7.5], [6.0, 8.5],  # Class 1
    [9.0, 2.0], [8.5, 1.5], [9.2, 2.8]   # Class 2
])
y_toy = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
N_toy = len(y_toy)
K_toy = 3

print(f"Number of samples N = {N_toy}, Number of classes K = {K_toy}")

# Step 1: Initial sample weights
w = np.ones(N_toy) / N_toy
print("\n1. Initial Sample Weights w_i^(1):")
for i in range(N_toy):
    print(f"  Sample {i} (Class {y_toy[i]}): w_{i}^(1) = {w[i]:.4f}")

# Step 2: Fit weak learner (Decision Tree with max_depth=1)
stump = DecisionTreeClassifier(max_depth=1, random_state=42)
stump.fit(X_toy, y_toy, sample_weight=w)

# Step 3: Weak learner predictions
preds = stump.predict(X_toy)
incorrect = (preds != y_toy).astype(int)

print("\n2. Weak Learner Predictions & Misclassifications:")
for i in range(N_toy):
    status = "INCORRECT" if incorrect[i] else "CORRECT"
    print(f"  Sample {i}: True = {y_toy[i]}, Pred = {preds[i]} -> {status}")

# Step 4: Weighted error epsilon_t
eps = np.sum(w * incorrect) / np.sum(w)
print(f"\n3. Weighted Error epsilon_1 = {eps:.4f}")

# Step 5: Check condition epsilon_t < (K - 1) / K
random_guess_err = (K_toy - 1) / K_toy
print(f"   Random multiclass error threshold (K-1)/K = {random_guess_err:.4f}")
assert eps < random_guess_err, "Weak learner not better than random guess!"

# Step 6: Compute alpha_t = log((1 - eps)/eps) + log(K - 1)
alpha = np.log((1.0 - eps) / eps) + np.log(K_toy - 1)
print(f"4. Learner Weight alpha_1 = log((1 - {eps:.4f})/{eps:.4f}) + log({K_toy - 1}) = {alpha:.4f}")

# Step 7: Update and normalize weights
# w_new = w * exp(alpha * I[pred != y])
w_unnorm = w * np.exp(alpha * incorrect)
w_norm = w_unnorm / np.sum(w_unnorm)

print("\n5. Updated Weights:")
print(f"{'Sample':<8} | {'True y':<8} | {'Pred h':<8} | {'w_old':<10} | {'Unnormalized':<14} | {'w_new (Normalized)':<18}")
print("-" * 75)
for i in range(N_toy):
    print(f"{i:<8} | {y_toy[i]:<8} | {preds[i]:<8} | {w[i]:<10.4f} | {w_unnorm[i]:<14.4f} | {w_norm[i]:<18.4f}")

print(f"\nSum of new weights = {np.sum(w_norm):.6f}")
