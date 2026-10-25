import numpy as np
from sklearn.tree import DecisionTreeClassifier

data = np.load("split_data.npz")
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

SEED = 26002

max_depths = [2, 4, 8, 16, None]
min_samples_leafs = [1, 2, 4]

print(f"{'max_depth':<10} | {'min_leaf':<10} | {'Train Acc':<10} | {'Val Acc':<10} | {'Test Acc':<10} | {'Leaves':<8} | {'Mean Samp/Leaf':<16} | {'Frac 1-Sample Leaf':<20}")
print("-" * 105)

results = []

for md in max_depths:
    for msl in min_samples_leafs:
        # DecisionTreeClassifier
        clf = DecisionTreeClassifier(max_depth=md, min_samples_leaf=msl, random_state=SEED)
        clf.fit(X_train, y_train)
        
        train_acc = clf.score(X_train, y_train)
        val_acc = clf.score(X_val, y_val)
        test_acc = clf.score(X_test, y_test)
        
        tree = clf.tree_
        # Leaves have children_left == -1
        leaf_mask = (tree.children_left == -1)
        n_leaves = tree.n_leaves # or np.sum(leaf_mask)
        leaf_samples = tree.n_node_samples[leaf_mask]
        mean_samples_leaf = np.mean(leaf_samples)
        frac_one_sample = np.mean(leaf_samples == 1)
        
        md_str = str(md) if md is not None else "None"
        print(f"{md_str:<10} | {msl:<10} | {train_acc:<10.4f} | {val_acc:<10.4f} | {test_acc:<10.4f} | {n_leaves:<8} | {mean_samples_leaf:<16.2f} | {frac_one_sample:<20.4f}")
        
        results.append({
            'max_depth': md,
            'min_samples_leaf': msl,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'n_leaves': n_leaves,
            'mean_samples_leaf': mean_samples_leaf,
            'frac_one_sample': frac_one_sample,
            'clf': clf
        })

# Find best by val accuracy
best_result = max(results, key=lambda r: (r['val_acc'], -r['min_samples_leaf'] if r['max_depth'] is not None else 0))
print("\nBest setting based on Validation Accuracy:")
print(f"max_depth = {best_result['max_depth']}, min_samples_leaf = {best_result['min_samples_leaf']}")
print(f"Train Acc: {best_result['train_acc']:.4f}, Val Acc: {best_result['val_acc']:.4f}, Test Acc: {best_result['test_acc']:.4f}")
print(f"Number of leaves: {best_result['n_leaves']}, Mean samples per leaf: {best_result['mean_samples_leaf']:.2f}, Frac 1-sample leaves: {best_result['frac_one_sample']:.4f}")
