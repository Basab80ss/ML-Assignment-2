import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Load split
data = np.load("split_data.npz")
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']
X_test_masked = data['X_test_masked']

SEED = 26002
rng = np.random.default_rng(SEED)

N_train = len(X_train)
B_values = [1, 5, 10, 25, 50, 100, 200]
max_B = max(B_values)

# Frozen tree setting from part (a)
frozen_max_depth = None
frozen_min_samples_leaf = 1

# Pre-generate 200 bootstrap samples and tree seeds
bootstrap_indices = []
tree_seeds = rng.integers(0, 1_000_000, size=max_B)

for b in range(max_B):
    boot = rng.choice(N_train, size=N_train, replace=True)
    bootstrap_indices.append(boot)

# Fit all 200 trees
trees = []
print(f"Training {max_B} trees with max_depth={frozen_max_depth}, min_samples_leaf={frozen_min_samples_leaf}...")
for b in range(max_B):
    clf = DecisionTreeClassifier(max_depth=frozen_max_depth, 
                                 min_samples_leaf=frozen_min_samples_leaf, 
                                 random_state=int(tree_seeds[b]))
    boot_idx = bootstrap_indices[b]
    clf.fit(X_train[boot_idx], y_train[boot_idx])
    trees.append(clf)

print("All trees trained.")

# Pre-compute predictions for all trees on train, val, test, test_masked
train_preds = np.array([t.predict(X_train) for t in trees]) # (200, 240)
val_preds = np.array([t.predict(X_val) for t in trees])     # (200, 80)
test_preds = np.array([t.predict(X_test) for t in trees])   # (200, 80)
test_masked_preds = np.array([t.predict(X_test_masked) for t in trees]) # (200, 80)

# Pre-compute class probabilities if needed
# Majority voting function
def ensemble_predict_and_margin(preds_subset, num_classes=40):
    # preds_subset shape: (B, M)
    B, M = preds_subset.shape
    final_preds = np.zeros(M, dtype=int)
    margins = np.zeros(M)
    winning_fractions = np.zeros(M)
    all_vote_counts = np.zeros((M, num_classes), dtype=int)
    
    for i in range(M):
        votes = np.bincount(preds_subset[:, i], minlength=num_classes)
        all_vote_counts[i] = votes
        sorted_votes = np.sort(votes)[::-1]
        winner = np.argmax(votes)
        final_preds[i] = winner
        winning_fractions[i] = sorted_votes[0] / B
        runner_up = sorted_votes[1] / B if len(sorted_votes) > 1 else 0.0
        margins[i] = winning_fractions[i] - runner_up
        
    return final_preds, winning_fractions, margins, all_vote_counts

# Theoretical expectation of distinct samples in bootstrap of size N
theoretical_distinct = N_train * (1.0 - (1.0 - 1.0/N_train)**N_train)
theoretical_frac = theoretical_distinct / N_train

print(f"\nTheoretical expected distinct samples: {theoretical_distinct:.2f} ({theoretical_frac*100:.2f}%)")

print("\n--- Question 1(b) and 1(c) Results for each B ---")
metrics_table = []

for B in B_values:
    # 1. Distinct examples average over B bootstrap samples
    distinct_counts = [len(np.unique(bootstrap_indices[b])) for b in range(B)]
    avg_distinct = np.mean(distinct_counts)
    
    # 2. Accuracy on train, val, test
    ens_train_pred, _, _, _ = ensemble_predict_and_margin(train_preds[:B])
    ens_val_pred, val_win_frac, val_margin, _ = ensemble_predict_and_margin(val_preds[:B])
    ens_test_pred, test_win_frac, _, _ = ensemble_predict_and_margin(test_preds[:B])
    
    train_acc = np.mean(ens_train_pred == y_train)
    val_acc = np.mean(ens_val_pred == y_val)
    test_acc = np.mean(ens_test_pred == y_test)
    
    # 3. OOB Accuracy
    # For each training sample i, find trees where i was NOT in bootstrap
    oob_correct = 0
    oob_total = 0
    for i in range(N_train):
        oob_trees = [b for b in range(B) if i not in bootstrap_indices[b]]
        if len(oob_trees) > 0:
            votes = np.bincount(train_preds[oob_trees, i], minlength=40)
            oob_pred = np.argmax(votes)
            if oob_pred == y_train[i]:
                oob_correct += 1
            oob_total += 1
    oob_acc = oob_correct / oob_total if oob_total > 0 else 0.0
    
    # 4. Average pairwise disagreement on validation set
    if B > 1:
        disagreements = []
        for j in range(B):
            for k in range(j+1, B):
                d = np.mean(val_preds[j] != val_preds[k])
                disagreements.append(d)
        avg_disagreement = np.mean(disagreements)
    else:
        avg_disagreement = 0.0
        
    # 5. Mean vote margin on validation set
    mean_margin = np.mean(val_margin)
    
    metrics_table.append({
        'B': B,
        'avg_distinct': avg_distinct,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'oob_acc': oob_acc,
        'disagreement': avg_disagreement,
        'mean_margin': mean_margin
    })

print(f"{'B':<5} | {'Avg Dist':<10} | {'Train Acc':<10} | {'Val Acc':<10} | {'Test Acc':<10} | {'OOB Acc':<10} | {'Disagree':<10} | {'Vote Margin':<10}")
print("-" * 90)
for r in metrics_table:
    dis_str = f"{r['disagreement']:.4f}" if r['B'] > 1 else "N/A"
    print(f"{r['B']:<5} | {r['avg_distinct']:<10.2f} | {r['train_acc']:<10.4f} | {r['val_acc']:<10.4f} | {r['test_acc']:<10.4f} | {r['oob_acc']:<10.4f} | {dis_str:<10} | {r['mean_margin']:<10.4f}")

# Save metrics
import json
with open("q1_metrics.json", "w") as f:
    json.dump(metrics_table, f, indent=4)
