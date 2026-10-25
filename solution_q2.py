"""
Machine Learning Assignment 2 - Question 2 Complete Solution
Student Roll Number: m26cse002 -> Seed: 26002
Topic: Multiclass Boosting from Scratch (SAMME AdaBoost) on Olivetti Faces
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import time

# ==============================================================================
# QUESTION 2(a): TOY THREE-CLASS EXAMPLE LOGGING ONE SAMME ROUND
# ==============================================================================
print("=" * 80)
print("QUESTION 2(a): SAMME Round 1 Walkthrough on Toy 3-Class Dataset (N <= 10)")
print("=" * 80)

# Create a toy 3-class 2D dataset with N = 9 samples (3 per class)
X_toy = np.array([
    [1.0, 2.0], [1.5, 1.8], [1.2, 2.5],  # Class 0
    [5.0, 8.0], [5.5, 7.5], [6.0, 8.5],  # Class 1
    [9.0, 2.0], [8.5, 1.5], [9.2, 2.8]   # Class 2
])
y_toy = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
N_toy = len(y_toy)
K_toy = 3

print(f"Number of samples N = {N_toy}, Number of classes K = {K_toy}\n")

# 1. Initial sample weights
w_toy = np.ones(N_toy) / N_toy

# 2. Fit weak learner (Decision Tree with max_depth=1)
stump = DecisionTreeClassifier(max_depth=1, random_state=42)
stump.fit(X_toy, y_toy, sample_weight=w_toy)

# 3. Weak learner predictions
preds_toy = stump.predict(X_toy)
incorrect_toy = (preds_toy != y_toy).astype(int)

# 4. Weighted error epsilon_t
eps_toy = np.sum(w_toy * incorrect_toy) / np.sum(w_toy)

# 5. Check random guess threshold (K - 1) / K
rand_thresh_toy = (K_toy - 1) / K_toy
assert eps_toy < rand_thresh_toy, "Weak learner is not better than random guessing!"

# 6. Learner weight alpha_t = log((1 - eps)/eps) + log(K - 1)
alpha_toy = np.log((1.0 - eps_toy) / eps_toy) + np.log(K_toy - 1)

# 7. Update and normalize weights
# w_new = w * exp(alpha * I[pred != y])
w_unnorm_toy = w_toy * np.exp(alpha_toy * incorrect_toy)
w_norm_toy = w_unnorm_toy / np.sum(w_unnorm_toy)

print(f"SAMME Step-by-Step Values for Round 1:")
print(f"  Initial weights:                 w_i^(1) = 1/{N_toy} = {1.0/N_toy:.4f} for all i")
print(f"  Weighted Error:                  epsilon_1 = {eps_toy:.4f} (Threshold (K-1)/K = {rand_thresh_toy:.4f})")
print(f"  Learner Weight:                  alpha_1 = log((1 - {eps_toy:.4f})/{eps_toy:.4f}) + log({K_toy - 1}) = {alpha_toy:.4f}")
print(f"  Weight multiplier for mistakes:  exp(alpha_1) = {np.exp(alpha_toy):.4f}")
print(f"  Weight multiplier for correct:   exp(0) = 1.0000\n")

print(f"{'Sample i':<10} | {'True y_i':<10} | {'Pred h_1(x_i)':<14} | {'w_i^(1)':<10} | {'Unnorm w_i^(2)':<16} | {'Norm w_i^(2)':<14}")
print("-" * 82)
for i in range(N_toy):
    print(f"{i:<10} | {y_toy[i]:<10} | {preds_toy[i]:<14} | {w_toy[i]:<10.4f} | {w_unnorm_toy[i]:<16.4f} | {w_norm_toy[i]:<14.4f}")
print(f"\nSum of normalized weights w_i^(2) = {np.sum(w_norm_toy):.6f}\n")


import os
os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Load saved stratified data split
split_file = "data/split_data.npz" if os.path.exists("data/split_data.npz") else "split_data.npz"
split_data = np.load(split_file)
X_train, y_train = split_data['X_train'], split_data['y_train']
X_val, y_val = split_data['X_val'], split_data['y_val']
X_test, y_test = split_data['X_test'], split_data['y_test']
X_test_masked = split_data['X_test_masked']

SEED = 26002
K = 40
N_train = len(X_train)

class SAMME_AdaBoost:
    def __init__(self, base_depth=4, n_rounds=100, random_state=SEED):
        self.base_depth = base_depth
        self.n_rounds = n_rounds
        self.random_state = random_state
        self.learners = []
        self.alphas = []
        self.weighted_errors = []
        self.max_sample_weights = []
        self.top10_weights = []
        self.sample_weights_history = []
        self.sample_predictions_history = []
        
    def fit(self, X, y):
        N = len(X)
        w = np.ones(N) / N
        rng = np.random.default_rng(self.random_state)
        random_error_threshold = (K - 1.0) / K  # 39/40 = 0.9750
        
        for t in range(self.n_rounds):
            tree_seed = int(rng.integers(0, 1_000_000))
            clf = DecisionTreeClassifier(max_depth=self.base_depth, random_state=tree_seed)
            clf.fit(X, y, sample_weight=w)
            
            preds = clf.predict(X)
            incorrect = (preds != y).astype(float)
            
            # Weighted error
            eps_t = np.sum(w * incorrect) / np.sum(w)
            
            # Edge case handling
            if eps_t <= 1e-10:
                alpha_t = np.log((1.0 - 1e-10) / 1e-10) + np.log(K - 1)
                self.learners.append(clf)
                self.alphas.append(alpha_t)
                self.weighted_errors.append(eps_t)
                self.max_sample_weights.append(np.max(w))
                self.top10_weights.append(np.sum(np.sort(w)[-10:]))
                self.sample_weights_history.append(w.copy())
                self.sample_predictions_history.append(preds.copy())
                break
                
            if eps_t >= random_error_threshold:
                print(f"Round {t+1}: Weak learner error {eps_t:.4f} >= {random_error_threshold:.4f}. Stopping.")
                break
                
            # SAMME alpha update rule: log((1 - eps_t)/eps_t) + log(K - 1)
            alpha_t = np.log((1.0 - eps_t) / eps_t) + np.log(K - 1)
            
            self.learners.append(clf)
            self.alphas.append(alpha_t)
            self.weighted_errors.append(eps_t)
            self.max_sample_weights.append(np.max(w))
            self.top10_weights.append(np.sum(np.sort(w)[-10:]))
            self.sample_weights_history.append(w.copy())
            self.sample_predictions_history.append(preds.copy())
            
            # Normalized sample weight update rule: w_i * exp(alpha_t * 1[h_t(x_i) != y_i])
            w = w * np.exp(alpha_t * incorrect)
            w = w / np.sum(w)
            
        self.final_weights = w.copy()
        self.sample_predictions_history = np.array(self.sample_predictions_history)
        self.sample_weights_history = np.array(self.sample_weights_history)
        return self

    def predict_scores(self, X, n_used=None):
        if n_used is None:
            n_used = len(self.learners)
        N = len(X)
        S = np.zeros((N, K))
        for t in range(n_used):
            preds = self.learners[t].predict(X)
            alpha = self.alphas[t]
            for i in range(N):
                S[i, preds[i]] += alpha
        return S

    def predict(self, X, n_used=None):
        S = self.predict_scores(X, n_used)
        return np.argmax(S, axis=1)

    def predict_confidence(self, X, n_used=None):
        if n_used is None:
            n_used = len(self.learners)
        S = self.predict_scores(X, n_used)
        sum_alpha = np.sum(self.alphas[:n_used])
        q = S / sum_alpha
        return np.max(q, axis=1)

    def score(self, X, y, n_used=None):
        preds = self.predict(X, n_used)
        return np.mean(preds == y)


# ==============================================================================
# QUESTION 2(b): WEAK LEARNER DEPTH & ROUNDS TUNING (CLEAN VAL ACCURACY ONLY)
# ==============================================================================
print("=" * 80)
print("QUESTION 2(b): SAMME Grid Search over Depths {1, 2, 3, 4} and Rounds {10, 25, 50, 100}")
print("=" * 80)

depths = [1, 2, 3, 4]
rounds_list = [10, 25, 50, 100]

print(f"{'Depth':<8} | {'Rounds':<8} | {'Train Acc':<10} | {'Val Acc':<10} | {'Test Acc':<10}")
print("-" * 54)

grid_b_results = []
trained_models = {}

for d in depths:
    model_d = SAMME_AdaBoost(base_depth=d, n_rounds=100, random_state=SEED)
    model_d.fit(X_train, y_train)
    trained_models[d] = model_d
    for r in rounds_list:
        tr_acc = model_d.score(X_train, y_train, n_used=r)
        v_acc = model_d.score(X_val, y_val, n_used=r)
        te_acc = model_d.score(X_test, y_test, n_used=r)
        grid_b_results.append({
            'depth': d,
            'rounds': r,
            'train_acc': tr_acc,
            'val_acc': v_acc,
            'test_acc': te_acc
        })
        print(f"{d:<8} | {r:<8} | {tr_acc:<10.4f} | {v_acc:<10.4f} | {te_acc:<10.4f}")

# Select best model using clean validation accuracy only
best_boost = max(grid_b_results, key=lambda x: x['val_acc'])
selected_depth = best_boost['depth']
selected_rounds = best_boost['rounds']

print(f"\n--> Selected Boosting Model (Tuned strictly via Validation Accuracy):")
print(f"    Weak Learner Depth: {selected_depth}, Boosting Rounds: {selected_rounds}")
print(f"    Train Acc: {best_boost['train_acc']:.4f} | Val Acc: {best_boost['val_acc']:.4f} | Test Acc: {best_boost['test_acc']:.4f}\n")

# Diagnostics Plot for Selected Model (Depth 4, 100 Rounds)
selected_model = trained_models[selected_depth]
rounds_axis = np.arange(1, len(selected_model.learners) + 1)

fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

# 1. Weighted Training Error
axes[0, 0].plot(rounds_axis, selected_model.weighted_errors, color='#1f77b4', lw=2)
axes[0, 0].axhline(y=(K-1)/K, color='r', linestyle='--', label=f'Random Guess Limit ({(K-1)/K:.3f})')
axes[0, 0].set_xlabel('Round $t$', fontsize=11)
axes[0, 0].set_ylabel(r'Weighted Training Error $\epsilon_t$', fontsize=11)
axes[0, 0].set_title(r'Weighted Training Error $\epsilon_t$ over Rounds', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, linestyle='--', alpha=0.6)
axes[0, 0].legend()

# 2. Learner Weight alpha_t
axes[0, 1].plot(rounds_axis, selected_model.alphas, color='#2ca02c', lw=2)
axes[0, 1].set_xlabel('Round $t$', fontsize=11)
axes[0, 1].set_ylabel(r'Learner Weight $\alpha_t$', fontsize=11)
axes[0, 1].set_title(r'Learner Weight $\alpha_t$ over Rounds', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, linestyle='--', alpha=0.6)

# 3. Maximum Sample Weight
axes[1, 0].plot(rounds_axis, selected_model.max_sample_weights, color='#d62728', lw=2)
axes[1, 0].axhline(y=1.0/N_train, color='gray', linestyle=':', label=f'Initial Weight (1/240 = {1.0/N_train:.4f})')
axes[1, 0].set_xlabel('Round $t$', fontsize=11)
axes[1, 0].set_ylabel(r'Maximum Sample Weight $\max_i w_i^{(t)}$', fontsize=11)
axes[1, 0].set_title(r'Maximum Sample Weight over Rounds', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, linestyle='--', alpha=0.6)
axes[1, 0].legend()

# 4. Total Weight of Top 10 Heaviest Samples
axes[1, 1].plot(rounds_axis, selected_model.top10_weights, color='#9467bd', lw=2)
axes[1, 1].axhline(y=10.0/N_train, color='gray', linestyle=':', label=f'Initial Top 10 Weight (10/240 = {10.0/N_train:.4f})')
axes[1, 1].set_xlabel('Round $t$', fontsize=11)
axes[1, 1].set_ylabel('Total Weight of 10 Heaviest Samples', fontsize=11)
axes[1, 1].set_title('Total Weight Carried by 10 Heaviest Images', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, linestyle='--', alpha=0.6)
axes[1, 1].legend()

plt.suptitle(f'Question 2(b): SAMME Diagnostics over 100 Rounds (Depth {selected_depth})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('q2_b_diagnostics_plot.png', dpi=300)
plt.savefig('figures/q2_b_diagnostics_plot.png', dpi=300)
plt.close()
print("[Generated Plot] Saved SAMME diagnostics plot as 'figures/q2_b_diagnostics_plot.png'\n")


# ==============================================================================
# QUESTION 2(c): BOOSTING VS BAGGING COMPARISON & HEAVIEST SAMPLE ANALYSIS
# ==============================================================================
print("=" * 80)
print("QUESTION 2(c): Performance & Training Time Comparison (25, 50, 100 Learners)")
print("=" * 80)

# Bagging Models (using frozen tree: max_depth=None, min_samples_leaf=1)
bag_compare_results = []
rng_bag = np.random.default_rng(SEED)

for B in [25, 50, 100]:
    t0 = time.time()
    trees = []
    tree_seeds = rng_bag.integers(0, 1_000_000, size=B)
    for b in range(B):
        boot = rng_bag.choice(N_train, size=N_train, replace=True)
        t = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, random_state=int(tree_seeds[b]))
        t.fit(X_train[boot], y_train[boot])
        trees.append(t)
    bag_time = time.time() - t0
    
    def predict_bag(t_list, X):
        preds = np.array([tree.predict(X) for tree in t_list])
        maj = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            maj[i] = np.argmax(np.bincount(preds[:, i], minlength=K))
        return maj

    tr_acc = np.mean(predict_bag(trees, X_train) == y_train)
    v_acc = np.mean(predict_bag(trees, X_val) == y_val)
    te_acc = np.mean(predict_bag(trees, X_test) == y_test)
    
    bag_compare_results.append({
        'learners': B,
        'train_acc': tr_acc,
        'val_acc': v_acc,
        'test_acc': te_acc,
        'time': bag_time
    })

# Boosting Models (Depth 4)
boost_compare_results = []
for R in [25, 50, 100]:
    t0 = time.time()
    bm = SAMME_AdaBoost(base_depth=4, n_rounds=R, random_state=SEED)
    bm.fit(X_train, y_train)
    boost_time = time.time() - t0
    
    tr_acc = bm.score(X_train, y_train)
    v_acc = bm.score(X_val, y_val)
    te_acc = bm.score(X_test, y_test)
    
    boost_compare_results.append({
        'learners': R,
        'train_acc': tr_acc,
        'val_acc': v_acc,
        'test_acc': te_acc,
        'time': boost_time
    })

print(f"{'Method':<10} | {'Learners':<10} | {'Train Acc':<10} | {'Val Acc':<10} | {'Clean Test Acc':<16} | {'Train Time (s)':<14}")
print("-" * 76)
for r in bag_compare_results:
    print(f"{'Bagging':<10} | {r['learners']:<10} | {r['train_acc']:<10.4f} | {r['val_acc']:<10.4f} | {r['test_acc']:<16.4f} | {r['time']:<14.2f}")
for r in boost_compare_results:
    print(f"{'Boosting':<10} | {r['learners']:<10} | {r['train_acc']:<10.4f} | {r['val_acc']:<10.4f} | {r['test_acc']:<16.4f} | {r['time']:<14.2f}")

# Inspect Heaviest Training Sample in Boosting
final_w = selected_model.sample_weights_history[-1]
heaviest_idx = np.argmax(final_w)
heaviest_weight = final_w[heaviest_idx]
heaviest_true_label = y_train[heaviest_idx]
heaviest_preds_hist = selected_model.sample_predictions_history[:, heaviest_idx]
mistakes_count = np.sum(heaviest_preds_hist != heaviest_true_label)
weight_traj = selected_model.sample_weights_history[:, heaviest_idx]

print(f"\nHeaviest Training Face Inspection:")
print(f"  Training Sample Index: {heaviest_idx}")
print(f"  True Identity:         Class {heaviest_true_label}")
print(f"  Final Weight:          {heaviest_weight:.6f} (baseline was {1.0/N_train:.6f}, {heaviest_weight / (1.0/N_train):.1f}x higher!)")
print(f"  Misclassified:         {mistakes_count} out of 100 rounds ({mistakes_count}% error across weak learners)")

# Plot heaviest face & weight trajectory
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
axes[0].imshow(X_train[heaviest_idx].reshape(64, 64), cmap='gray')
axes[0].set_title(f"Heaviest Training Image (Idx {heaviest_idx})\nTrue Identity: Class {heaviest_true_label} | Misclassified in {mistakes_count}/100 Rounds", fontsize=10)
axes[0].axis('off')

axes[1].plot(np.arange(1, 101), weight_traj, color='#d62728', lw=2)
axes[1].axhline(y=1.0/N_train, color='gray', linestyle=':', label=f'Initial Uniform Weight ({1.0/N_train:.4f})')
axes[1].set_xlabel('Boosting Round $t$', fontsize=11)
axes[1].set_ylabel('Sample Weight $w_i^{(t)}$', fontsize=11)
axes[1].set_title(f'Weight Evolution of Sample {heaviest_idx}', fontsize=12, fontweight='bold')
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend()

plt.tight_layout()
plt.savefig('q2_c_heaviest_face.png', dpi=300)
plt.savefig('figures/q2_c_heaviest_face.png', dpi=300)
plt.close()
print("[Generated Plot] Saved heaviest face plot as 'figures/q2_c_heaviest_face.png'\n")


# ==============================================================================
# QUESTION 2(d): CLEAN VS MASKED TEST EVALUATION & DIFFERENTIAL RESPONSE CASE
# ==============================================================================
print("=" * 80)
print("QUESTION 2(d): Clean vs Masked Test Set Comparison & Occlusion Case Study")
print("=" * 80)

# 1. Single Tree Baseline
single_tree = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, random_state=SEED)
single_tree.fit(X_train, y_train)

st_cl_acc = np.mean(single_tree.predict(X_test) == y_test)
st_cl_conf = np.mean(np.max(single_tree.predict_proba(X_test), axis=1))
st_mk_acc = np.mean(single_tree.predict(X_test_masked) == y_test)
st_mk_conf = np.mean(np.max(single_tree.predict_proba(X_test_masked), axis=1))

# 2. Final Bagged Model (B=100)
bag_trees_100 = []
rng_bag100 = np.random.default_rng(SEED)
tree_seeds100 = rng_bag100.integers(0, 1_000_000, size=100)
for b in range(100):
    boot = rng_bag100.choice(N_train, size=N_train, replace=True)
    t = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, random_state=int(tree_seeds100[b]))
    t.fit(X_train[boot], y_train[boot])
    bag_trees_100.append(t)

bag_cl_preds = np.array([t.predict(X_test) for t in bag_trees_100])
bag_mk_preds = np.array([t.predict(X_test_masked) for t in bag_trees_100])

bag_cl_maj = np.zeros(len(y_test), dtype=int)
bag_cl_conf_arr = np.zeros(len(y_test))
bag_mk_maj = np.zeros(len(y_test), dtype=int)
bag_mk_conf_arr = np.zeros(len(y_test))

for i in range(len(y_test)):
    v_cl = np.bincount(bag_cl_preds[:, i], minlength=K)
    bag_cl_maj[i] = np.argmax(v_cl)
    bag_cl_conf_arr[i] = np.max(v_cl) / 100.0
    
    v_mk = np.bincount(bag_mk_preds[:, i], minlength=K)
    bag_mk_maj[i] = np.argmax(v_mk)
    bag_mk_conf_arr[i] = np.max(v_mk) / 100.0

bag_cl_acc = np.mean(bag_cl_maj == y_test)
bag_cl_conf = np.mean(bag_cl_conf_arr)
bag_mk_acc = np.mean(bag_mk_maj == y_test)
bag_mk_conf = np.mean(bag_mk_conf_arr)

# 3. Selected Boosting Model (Depth 4, 100 rounds)
boost_cl_pred = selected_model.predict(X_test)
boost_cl_acc = np.mean(boost_cl_pred == y_test)
boost_cl_conf = np.mean(selected_model.predict_confidence(X_test))

boost_mk_pred = selected_model.predict(X_test_masked)
boost_mk_acc = np.mean(boost_mk_pred == y_test)
boost_mk_conf = np.mean(selected_model.predict_confidence(X_test_masked))

print(f"{'Model':<25} | {'Clean Acc':<12} | {'Clean Conf':<12} | {'Masked Acc':<12} | {'Masked Conf':<12}")
print("-" * 81)
print(f"{'Single Tree Baseline':<25} | {st_cl_acc:<12.4f} | {st_cl_conf:<12.4f} | {st_mk_acc:<12.4f} | {st_mk_conf:<12.4f}")
print(f"{'Bagging (B=100)':<25} | {bag_cl_acc:<12.4f} | {bag_cl_conf:<12.4f} | {bag_mk_acc:<12.4f} | {bag_mk_conf:<12.4f}")
print(f"{'Boosting (Depth 4, 100 R)':<25} | {boost_cl_acc:<12.4f} | {boost_cl_conf:<12.4f} | {boost_mk_acc:<12.4f} | {boost_mk_conf:<12.4f}")

# Find face with differential response: Both correct on Clean, but Bagging survives and Boosting fails on Masked
diff_cases = [i for i in range(len(y_test)) if bag_cl_maj[i] == y_test[i] and boost_cl_pred[i] == y_test[i] and bag_mk_maj[i] == y_test[i] and boost_mk_pred[i] != y_test[i]]

diff_face_idx = diff_cases[0] # Test Index 57 (Class 28)
print(f"\nSelected Face with Differential Response to Masking: Test Index {diff_face_idx}")
print(f"  True Identity:                  Class {y_test[diff_face_idx]}")
print(f"  Clean Test Prediction:")
print(f"    - Bagging:                    Class {bag_cl_maj[diff_face_idx]} (Confidence: {bag_cl_conf_arr[diff_face_idx]:.2f}) -> CORRECT")
print(f"    - Boosting:                   Class {boost_cl_pred[diff_face_idx]} (Confidence: {selected_model.predict_confidence(X_test)[diff_face_idx]:.2f}) -> CORRECT")
print(f"  Masked Test Prediction (16x16 Center Masked):")
print(f"    - Bagging:                    Class {bag_mk_maj[diff_face_idx]} (Confidence: {bag_mk_conf_arr[diff_face_idx]:.2f}) -> CORRECT (Robust!)")
print(f"    - Boosting:                   Class {boost_mk_pred[diff_face_idx]} (Confidence: {selected_model.predict_confidence(X_test_masked)[diff_face_idx]:.2f}) -> WRONG (Failed!)")

# Plot differential response
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
axes[0].imshow(X_test[diff_face_idx].reshape(64, 64), cmap='gray')
axes[0].set_title(f"Clean Face (Test Idx {diff_face_idx}, True Class {y_test[diff_face_idx]})\nBagging: Class {bag_cl_maj[diff_face_idx]} (Correct) | Boosting: Class {boost_cl_pred[diff_face_idx]} (Correct)", fontsize=10)
axes[0].axis('off')

axes[1].imshow(X_test_masked[diff_face_idx].reshape(64, 64), cmap='gray')
axes[1].set_title(f"Masked Face (Center 16x16 Zeroed)\nBagging: Class {bag_mk_maj[diff_face_idx]} (CORRECT) | Boosting: Class {boost_mk_pred[diff_face_idx]} (WRONG)", fontsize=10)
axes[1].axis('off')

plt.suptitle('Question 2(d): Differential Response to Center Occlusion', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('q2_d_robustness_face.png', dpi=300)
plt.savefig('figures/q2_d_robustness_face.png', dpi=300)
plt.close()
print("[Generated Plot] Saved differential response plot as 'figures/q2_d_robustness_face.png'")
print("\n========================= Question 2 Execution Complete =========================")
