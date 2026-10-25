"""
Machine Learning Assignment 2 - Question 1 Complete Solution
Student Roll Number: m26cse002 -> Seed: 26002
Topic: Decision Tree Baseline and Bagging from Scratch on Olivetti Faces
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier

# ==============================================================================
# 0. SETUP & DATASET PREPARATION
# ==============================================================================
ROLL_NO = "m26cse002"
# Extract numeric characters in order
SEED = int("".join([c for c in ROLL_NO if c.isdigit()]))
print(f"==================================================================")
print(f"ML Assignment 2 - Question 1 Solution")
print(f"Roll Number: {ROLL_NO} | Extracted Random Seed: {SEED}")
print(f"==================================================================\n")

# Load 400 faces (40 identities, 10 images each, 64x64)
faces = fetch_olivetti_faces(shuffle=False)
X_all = faces.data   # shape (400, 4096)
y_all = faces.target # shape (400,)

rng = np.random.default_rng(SEED)

# Stratified split: within each identity, 6 train, 2 val, 2 test
train_idx, val_idx, test_idx = [], [], []
for identity in range(40):
    id_indices = np.where(y_all == identity)[0]
    shuffled = rng.permutation(id_indices)
    train_idx.extend(shuffled[:6])
    val_idx.extend(shuffled[6:8])
    test_idx.extend(shuffled[8:])

train_idx = np.array(train_idx)
val_idx = np.array(val_idx)
test_idx = np.array(test_idx)

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_val, y_val = X_all[val_idx], y_all[val_idx]
X_test, y_test = X_all[test_idx], y_all[test_idx]

# Robustness experiment: Masked test set (16x16 center block rows & cols 24 to 39 set to 0)
X_test_masked = X_test.copy()
for i in range(len(X_test_masked)):
    img = X_test_masked[i].reshape(64, 64)
    img[24:40, 24:40] = 0.0
    X_test_masked[i] = img.flatten()

import os
os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# Save split for re-use across the assignment
np.savez("split_data.npz", 
         train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
         X_train=X_train, y_train=y_train,
         X_val=X_val, y_val=y_val,
         X_test=X_test, y_test=y_test,
         X_test_masked=X_test_masked)
np.savez("data/split_data.npz", 
         train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
         X_train=X_train, y_train=y_train,
         X_val=X_val, y_val=y_val,
         X_test=X_test, y_test=y_test,
         X_test_masked=X_test_masked)

print(f"Dataset Split Completed:")
print(f"  Training set:   {X_train.shape[0]} samples (6 per identity)")
print(f"  Validation set: {X_val.shape[0]} samples (2 per identity)")
print(f"  Test set:       {X_test.shape[0]} samples (2 per identity)")
print(f"  Features:       {X_train.shape[1]} raw flattened pixels\n")


# ==============================================================================
# QUESTION 1(a): SINGLE TREE BASELINE & HYPERPARAMETER EXPLORATION
# ==============================================================================
print("=" * 80)
print("QUESTION 1(a): Decision Tree Baseline Grid Search")
print("=" * 80)

max_depths = [2, 4, 8, 16, None]
min_samples_leafs = [1, 2, 4]

print(f"{'max_depth':<10} | {'min_leaf':<10} | {'Train Acc':<10} | {'Val Acc':<10} | {'Test Acc':<10} | {'Leaves':<8} | {'Mean Samp/Leaf':<16} | {'Frac 1-Sample Leaf':<18}")
print("-" * 105)

grid_results = []
for md in max_depths:
    for msl in min_samples_leafs:
        clf = DecisionTreeClassifier(max_depth=md, min_samples_leaf=msl, random_state=SEED)
        clf.fit(X_train, y_train)
        
        train_acc = clf.score(X_train, y_train)
        val_acc = clf.score(X_val, y_val)
        test_acc = clf.score(X_test, y_test)
        
        tree = clf.tree_
        leaf_mask = (tree.children_left == -1)
        n_leaves = tree.n_leaves
        leaf_samples = tree.n_node_samples[leaf_mask]
        mean_samples = np.mean(leaf_samples)
        frac_singleton = np.mean(leaf_samples == 1)
        
        md_str = str(md) if md is not None else "None"
        print(f"{md_str:<10} | {msl:<10} | {train_acc:<10.4f} | {val_acc:<10.4f} | {test_acc:<10.4f} | {n_leaves:<8} | {mean_samples:<16.2f} | {frac_singleton:<18.4f}")
        
        grid_results.append({
            'max_depth': md,
            'min_samples_leaf': msl,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'n_leaves': n_leaves,
            'mean_samples': mean_samples,
            'frac_singleton': frac_singleton,
            'clf': clf
        })

# Select best setting using validation accuracy only
best_setting = max(grid_results, key=lambda r: (r['val_acc'], -r['min_samples_leaf'] if r['max_depth'] is not None else 0))
frozen_max_depth = best_setting['max_depth']
frozen_min_samples_leaf = best_setting['min_samples_leaf']
frozen_single_tree = best_setting['clf']

print(f"\n--> Frozen Single Tree Setting Selected by Validation Accuracy:")
print(f"    max_depth = {frozen_max_depth}, min_samples_leaf = {frozen_min_samples_leaf}")
print(f"    Train Acc: {best_setting['train_acc']:.4f} | Val Acc: {best_setting['val_acc']:.4f} | Test Acc: {best_setting['test_acc']:.4f}")
print(f"    Total Leaves: {best_setting['n_leaves']} | Mean Samples/Leaf: {best_setting['mean_samples']:.2f} | Singleton Leaves: {best_setting['frac_singleton']*100:.2f}%\n")


# ==============================================================================
# QUESTION 1(b) & 1(c): BAGGING FROM SCRATCH & ENSEMBLE METRICS
# ==============================================================================
print("=" * 80)
print("QUESTION 1(b) & 1(c): Bagging Implementation, Diversity & Ensemble Size")
print("=" * 80)

B_values = [1, 5, 10, 25, 50, 100, 200]
max_B = max(B_values)
N_train = len(X_train)

# Theoretical expectation of distinct samples in bootstrap sample of size N:
# E[distinct] = N * (1 - (1 - 1/N)^N)
theoretical_distinct = N_train * (1.0 - (1.0 - 1.0 / N_train)**N_train)
theoretical_pct = (theoretical_distinct / N_train) * 100.0

print(f"Theoretical Expectation of Distinct Samples in Bootstrap of size N={N_train}:")
print(f"  E[distinct] = {theoretical_distinct:.2f} samples ({theoretical_pct:.2f}% of training set)\n")

# Generate bootstrap samples and train ensemble
ensemble_rng = np.random.default_rng(SEED)
bootstrap_indices = []
tree_seeds = ensemble_rng.integers(0, 1_000_000, size=max_B)

bagged_trees = []
for b in range(max_B):
    boot = ensemble_rng.choice(N_train, size=N_train, replace=True)
    bootstrap_indices.append(boot)
    tree = DecisionTreeClassifier(max_depth=frozen_max_depth,
                                  min_samples_leaf=frozen_min_samples_leaf,
                                  random_state=int(tree_seeds[b]))
    tree.fit(X_train[boot], y_train[boot])
    bagged_trees.append(tree)

# Pre-compute individual tree predictions
all_train_preds = np.array([t.predict(X_train) for t in bagged_trees]) # (max_B, N_train)
all_val_preds = np.array([t.predict(X_val) for t in bagged_trees])     # (max_B, N_val)
all_test_preds = np.array([t.predict(X_test) for t in bagged_trees])   # (max_B, N_test)
all_masked_preds = np.array([t.predict(X_test_masked) for t in bagged_trees])

# Function to compute ensemble majority vote and vote margin
def get_ensemble_predictions(preds_matrix, num_classes=40):
    # preds_matrix: (B, M)
    B, M = preds_matrix.shape
    ens_preds = np.zeros(M, dtype=int)
    margins = np.zeros(M)
    winning_fracs = np.zeros(M)
    vote_matrix = np.zeros((M, num_classes), dtype=int)
    
    for i in range(M):
        votes = np.bincount(preds_matrix[:, i], minlength=num_classes)
        vote_matrix[i] = votes
        sorted_votes = np.sort(votes)[::-1]
        ens_preds[i] = np.argmax(votes)
        winning_fracs[i] = sorted_votes[0] / B
        runner_up_frac = sorted_votes[1] / B if len(sorted_votes) > 1 else 0.0
        margins[i] = winning_fracs[i] - runner_up_frac
        
    return ens_preds, winning_fracs, margins, vote_matrix

print(f"{'B':<5} | {'Avg Dist':<10} | {'Train Acc':<10} | {'Val Acc':<10} | {'Test Acc':<10} | {'OOB Acc':<10} | {'Disagree':<10} | {'Mean Margin':<12}")
print("-" * 92)

b_results = []
for B in B_values:
    # 1. Distinct samples
    distincts = [len(np.unique(bootstrap_indices[b])) for b in range(B)]
    avg_dist = np.mean(distincts)
    
    # 2. Accuracies
    ens_train_pred, _, _, _ = get_ensemble_predictions(all_train_preds[:B])
    ens_val_pred, _, val_margins, _ = get_ensemble_predictions(all_val_preds[:B])
    ens_test_pred, _, _, _ = get_ensemble_predictions(all_test_preds[:B])
    
    tr_acc = np.mean(ens_train_pred == y_train)
    v_acc = np.mean(ens_val_pred == y_val)
    te_acc = np.mean(ens_test_pred == y_test)
    
    # 3. Out-Of-Bag (OOB) Accuracy
    oob_correct = 0
    oob_eval_count = 0
    for i in range(N_train):
        oob_tree_indices = [b for b in range(B) if i not in bootstrap_indices[b]]
        if len(oob_tree_indices) > 0:
            oob_votes = np.bincount(all_train_preds[oob_tree_indices, i], minlength=40)
            if np.argmax(oob_votes) == y_train[i]:
                oob_correct += 1
            oob_eval_count += 1
    oob_acc = oob_correct / oob_eval_count if oob_eval_count > 0 else 0.0
    
    # 4. Average Pairwise Disagreement on validation set
    if B > 1:
        d_list = []
        for j in range(B):
            for k in range(j + 1, B):
                d_list.append(np.mean(all_val_preds[j] != all_val_preds[k]))
        pairwise_disagreement = np.mean(d_list)
    else:
        pairwise_disagreement = 0.0
        
    # 5. Mean Vote Margin on validation set
    mean_margin = np.mean(val_margins)
    
    dis_str = f"{pairwise_disagreement:.4f}" if B > 1 else "N/A"
    print(f"{B:<5} | {avg_dist:<10.2f} | {tr_acc:<10.4f} | {v_acc:<10.4f} | {te_acc:<10.4f} | {oob_acc:<10.4f} | {dis_str:<10} | {mean_margin:<12.4f}")
    
    b_results.append({
        'B': B,
        'avg_dist': avg_dist,
        'train_acc': tr_acc,
        'val_acc': v_acc,
        'test_acc': te_acc,
        'oob_acc': oob_acc,
        'disagreement': pairwise_disagreement,
        'mean_margin': mean_margin
    })

# Plot Q1(c) metrics
plt.figure(figsize=(9, 5.5))
plt.plot(B_values, [r['oob_acc'] for r in b_results], 'o-', label='Out-of-Bag (OOB) Accuracy', color='#1f77b4', lw=2)
plt.plot(B_values, [r['val_acc'] for r in b_results], 's--', label='Validation Accuracy', color='#2ca02c', lw=2)
plt.plot(B_values[1:], [r['disagreement'] for r in b_results][1:], '^-', label='Avg Pairwise Disagreement', color='#d62728', lw=2)
plt.plot(B_values, [r['mean_margin'] for r in b_results], 'd-', label='Mean Vote Margin', color='#9467bd', lw=2)

plt.xscale('log')
plt.xticks(B_values, [str(b) for b in B_values])
plt.xlabel('Ensemble Size $B$ (log scale)', fontsize=12)
plt.ylabel('Metric Value', fontsize=12)
plt.title('Question 1(c): Ensemble Metrics vs. Number of Trees $B$', fontsize=13, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10, loc='best')
plt.tight_layout()
plt.savefig('q1_c_metrics_plot.png', dpi=300)
plt.savefig('figures/q1_c_metrics_plot.png', dpi=300)
plt.close()
print(f"\n[Generated Plot] Saved ensemble metrics plot as 'figures/q1_c_metrics_plot.png'")


# ==============================================================================
# QUESTION 1(d): MODEL COMPARISON (CLEAN VS MASKED) & ERROR CASE ANALYSIS
# ==============================================================================
print("\n" + "=" * 80)
print("QUESTION 1(d): Robustness, Confidence & Case Study of Errors")
print("=" * 80)

FINAL_B = 100 # Selected based on validation plateau (88.75% val acc, fast inference)
print(f"Selected Final Ensemble Size: B = {FINAL_B}\n")

# 1. Single Tree Evaluation
st_clean_pred = frozen_single_tree.predict(X_test)
st_clean_acc = np.mean(st_clean_pred == y_test)
st_clean_probs = frozen_single_tree.predict_proba(X_test)
st_clean_conf = np.mean(np.max(st_clean_probs, axis=1))

st_masked_pred = frozen_single_tree.predict(X_test_masked)
st_masked_acc = np.mean(st_masked_pred == y_test)
st_masked_probs = frozen_single_tree.predict_proba(X_test_masked)
st_masked_conf = np.mean(np.max(st_masked_probs, axis=1))

# 2. Bagging Evaluation (B=100)
bag_clean_pred, bag_clean_conf_list, _, bag_clean_votes = get_ensemble_predictions(all_test_preds[:FINAL_B])
bag_clean_acc = np.mean(bag_clean_pred == y_test)
bag_clean_conf = np.mean(bag_clean_conf_list)

bag_masked_pred, bag_masked_conf_list, _, bag_masked_votes = get_ensemble_predictions(all_masked_preds[:FINAL_B])
bag_masked_acc = np.mean(bag_masked_pred == y_test)
bag_masked_conf = np.mean(bag_masked_conf_list)

print(f"{'Model':<22} | {'Clean Test Acc':<16} | {'Clean Mean Conf':<16} | {'Masked Test Acc':<16} | {'Masked Mean Conf':<16}")
print("-" * 96)
print(f"{'Single Tree Baseline':<22} | {st_clean_acc:<16.4f} | {st_clean_conf:<16.4f} | {st_masked_acc:<16.4f} | {st_masked_conf:<16.4f}")
print(f"{f'Bagging Ensemble (B={FINAL_B})':<22} | {bag_clean_acc:<16.4f} | {bag_clean_conf:<16.4f} | {bag_masked_acc:<16.4f} | {bag_masked_conf:<16.4f}")

# 3. Find Error Cases
corrects_idx = np.where((st_clean_pred != y_test) & (bag_clean_pred == y_test))[0]
both_wrong_idx = np.where((st_clean_pred != y_test) & (bag_clean_pred != y_test))[0]

print(f"\nError Analysis Summary on Clean Test Set:")
print(f"  Faces where Bagging corrects the Single Tree: {len(corrects_idx)} faces")
print(f"  Faces where Both models are wrong:            {len(both_wrong_idx)} faces")

selected_cases = [corrects_idx[0], corrects_idx[1], both_wrong_idx[0], both_wrong_idx[1]]
case_types = ["Bagging Corrects Tree", "Bagging Corrects Tree", "Both Wrong", "Both Wrong"]

print("\n--- Detailed Case Inspection ---")
for idx, ctype in zip(selected_cases, case_types):
    votes = bag_clean_votes[idx]
    top_c = np.argsort(votes)[::-1][:5]
    print(f"\nCase: {ctype} | Test Index: {idx}")
    print(f"  True Label:               Class {y_test[idx]}")
    print(f"  Single Tree Prediction:   Class {st_clean_pred[idx]}")
    print(f"  Bagging (B={FINAL_B}) Pred:   Class {bag_clean_pred[idx]} ({'CORRECT' if bag_clean_pred[idx] == y_test[idx] else 'WRONG'})")
    print(f"  Top-5 Bagging Votes (out of {FINAL_B}):")
    for c in top_c:
        print(f"    Class {c:>2}: {votes[c]:>2} votes ({votes[c]/FINAL_B*100:4.1f}%)")

# Plot the 4 faces and their vote distributions
fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
for col in range(4):
    idx = selected_cases[col]
    ctype = case_types[col]
    
    # Image
    axes[0, col].imshow(X_test[idx].reshape(64, 64), cmap='gray')
    axes[0, col].set_title(f"{ctype}\nIdx {idx} | True: C{y_test[idx]}\nTree: C{st_clean_pred[idx]} | Bag: C{bag_clean_pred[idx]}", fontsize=10)
    axes[0, col].axis('off')
    
    # Vote Distribution Bar Chart
    votes = bag_clean_votes[idx]
    top5 = np.argsort(votes)[::-1][:5]
    bar_colors = ['#2ca02c' if c == y_test[idx] else '#1f77b4' for c in top5]
    bars = axes[1, col].bar([f"C{c}" for c in top5], votes[top5], color=bar_colors)
    axes[1, col].set_ylabel(f'Votes (out of {FINAL_B})', fontsize=9)
    axes[1, col].set_ylim(0, FINAL_B)
    axes[1, col].grid(axis='y', linestyle='--', alpha=0.5)

plt.suptitle('Question 1(d): Error Analysis Cases (Images & Bagging Vote Distributions)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('q1_d_error_cases.png', dpi=300)
plt.savefig('figures/q1_d_error_cases.png', dpi=300)
plt.close()
print(f"\n[Generated Plot] Saved error cases plot as 'figures/q1_d_error_cases.png'")
print("\n========================= Execution Complete =========================")
