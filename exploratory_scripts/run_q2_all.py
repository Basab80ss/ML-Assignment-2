import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import time

# Load data split
data = np.load("split_data.npz")
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']
X_test_masked = data['X_test_masked']

SEED = 26002
K = 40
N_train = len(X_train)

# ==============================================================================
# SAMME Implementation
# ==============================================================================
class SAMME_Classifier:
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
        self.sample_predictions_history = [] # (T, N_train)
        
    def fit(self, X, y):
        N = len(X)
        w = np.ones(N) / N
        rng = np.random.default_rng(self.random_state)
        random_error_threshold = (K - 1.0) / K
        
        for t in range(self.n_rounds):
            tree_seed = int(rng.integers(0, 1_000_000))
            clf = DecisionTreeClassifier(max_depth=self.base_depth, random_state=tree_seed)
            clf.fit(X, y, sample_weight=w)
            
            preds = clf.predict(X)
            incorrect = (preds != y).astype(float)
            
            eps_t = np.sum(w * incorrect) / np.sum(w)
            
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
                
            alpha_t = np.log((1.0 - eps_t) / eps_t) + np.log(K - 1)
            
            self.learners.append(clf)
            self.alphas.append(alpha_t)
            self.weighted_errors.append(eps_t)
            self.max_sample_weights.append(np.max(w))
            self.top10_weights.append(np.sum(np.sort(w)[-10:]))
            self.sample_weights_history.append(w.copy())
            self.sample_predictions_history.append(preds.copy())
            
            # Update weights for next round
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
# 2(b) Train Selected Boosting Model (Depth 4, 100 rounds) and Plot Metrics
# ==============================================================================
print("\nTraining Selected SAMME Boosting Model (Depth 4, 100 rounds)...")
boost_model = SAMME_Classifier(base_depth=4, n_rounds=100, random_state=SEED)
boost_model.fit(X_train, y_train)

rounds_arr = np.arange(1, len(boost_model.learners) + 1)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# 1. Weighted Training Error
axes[0, 0].plot(rounds_arr, boost_model.weighted_errors, color='#1f77b4', lw=2)
axes[0, 0].axhline(y=(K-1)/K, color='r', linestyle='--', label=f'Random Guess Threshold ({(K-1)/K:.3f})')
axes[0, 0].set_xlabel('Round $t$', fontsize=11)
axes[0, 0].set_ylabel(r'Weighted Training Error $\epsilon_t$', fontsize=11)
axes[0, 0].set_title(r'Weighted Training Error $\epsilon_t$ over Rounds', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, linestyle='--', alpha=0.6)
axes[0, 0].legend()

# 2. Learner Weight alpha_t
axes[0, 1].plot(rounds_arr, boost_model.alphas, color='#2ca02c', lw=2)
axes[0, 1].set_xlabel('Round $t$', fontsize=11)
axes[0, 1].set_ylabel(r'Learner Weight $\alpha_t$', fontsize=11)
axes[0, 1].set_title(r'Learner Weight $\alpha_t$ over Rounds', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, linestyle='--', alpha=0.6)

# 3. Maximum Sample Weight
axes[1, 0].plot(rounds_arr, boost_model.max_sample_weights, color='#d62728', lw=2)
axes[1, 0].axhline(y=1.0/N_train, color='gray', linestyle=':', label=f'Initial Weight (1/{N_train} = {1.0/N_train:.4f})')
axes[1, 0].set_xlabel('Round $t$', fontsize=11)
axes[1, 0].set_ylabel(r'Maximum Sample Weight $\max_i w_i^{(t)}$', fontsize=11)
axes[1, 0].set_title(r'Maximum Sample Weight over Rounds', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, linestyle='--', alpha=0.6)
axes[1, 0].legend()

# 4. Total Weight of 10 Heaviest Samples
axes[1, 1].plot(rounds_arr, boost_model.top10_weights, color='#9467bd', lw=2)
axes[1, 1].axhline(y=10.0/N_train, color='gray', linestyle=':', label=f'Initial Weight (10/{N_train} = {10.0/N_train:.4f})')
axes[1, 1].set_xlabel('Round $t$', fontsize=11)
axes[1, 1].set_ylabel('Total Weight of Top 10 Samples', fontsize=11)
axes[1, 1].set_title('Total Weight Carried by Top 10 Heaviest Images', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, linestyle='--', alpha=0.6)
axes[1, 1].legend()

plt.suptitle('Question 2(b): SAMME Boosting Diagnostics over 100 Rounds (Depth 4)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('q2_b_diagnostics_plot.png', dpi=300)
plt.close()
print("Saved q2_b_diagnostics_plot.png")

# ==============================================================================
# 2(c) Compare Boosting vs Bagging (25, 50, 100 learners)
# ==============================================================================
print("\n--- Question 2(c): Comparing Boosting and Bagging (25, 50, 100 learners) ---")

# Train Bagging ensembles of 25, 50, 100 from scratch (frozen max_depth=None)
bagging_results = []
for B in [25, 50, 100]:
    t0 = time.time()
    rng_b = np.random.default_rng(SEED)
    b_trees = []
    tree_seeds = rng_b.integers(0, 1_000_000, size=B)
    for b in range(B):
        boot = rng_b.choice(N_train, size=N_train, replace=True)
        t = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, random_state=int(tree_seeds[b]))
        t.fit(X_train[boot], y_train[boot])
        b_trees.append(t)
    train_time = time.time() - t0
    
    # Evaluate
    def eval_bag(trees, X):
        preds = np.array([t.predict(X) for t in trees]) # (B, M)
        M = X.shape[0]
        maj = np.zeros(M, dtype=int)
        for i in range(M):
            maj[i] = np.argmax(np.bincount(preds[:, i], minlength=K))
        return maj
        
    tr_acc = np.mean(eval_bag(b_trees, X_train) == y_train)
    v_acc = np.mean(eval_bag(b_trees, X_val) == y_val)
    te_acc = np.mean(eval_bag(b_trees, X_test) == y_test)
    
    bagging_results.append({
        'learners': B,
        'train_acc': tr_acc,
        'val_acc': v_acc,
        'test_acc': te_acc,
        'time': train_time
    })

# Train Boosting models for 25, 50, 100 learners to measure exact training time
boosting_results = []
for R in [25, 50, 100]:
    t0 = time.time()
    bm = SAMME_Classifier(base_depth=4, n_rounds=R, random_state=SEED)
    bm.fit(X_train, y_train)
    b_time = time.time() - t0
    
    tr_acc = bm.score(X_train, y_train)
    v_acc = bm.score(X_val, y_val)
    te_acc = bm.score(X_test, y_test)
    
    boosting_results.append({
        'learners': R,
        'train_acc': tr_acc,
        'val_acc': v_acc,
        'test_acc': te_acc,
        'time': b_time
    })

print(f"{'Method':<10} | {'Learners':<10} | {'Train Acc':<10} | {'Val Acc':<10} | {'Test Acc':<10} | {'Train Time (s)':<15}")
print("-" * 75)
for r in bagging_results:
    print(f"{'Bagging':<10} | {r['learners']:<10} | {r['train_acc']:<10.4f} | {r['val_acc']:<10.4f} | {r['test_acc']:<10.4f} | {r['time']:<15.2f}")
for r in boosting_results:
    print(f"{'Boosting':<10} | {r['learners']:<10} | {r['train_acc']:<10.4f} | {r['val_acc']:<10.4f} | {r['test_acc']:<10.4f} | {r['time']:<15.2f}")

# ==============================================================================
# 2(c) Inspect Heaviest Training Sample in Boosting
# ==============================================================================
# Sample with highest final weight (or highest max weight over rounds)
final_weights = boost_model.sample_weights_history[-1]
heaviest_idx = np.argmax(final_weights)
heaviest_weight = final_weights[heaviest_idx]
heaviest_true_label = y_train[heaviest_idx]

# Check its prediction history across the 100 rounds
preds_history = boost_model.sample_predictions_history[:, heaviest_idx] # (100,)
mistakes_count = np.sum(preds_history != heaviest_true_label)
weight_trajectory = boost_model.sample_weights_history[:, heaviest_idx]

print(f"\nHeaviest Training Face Analysis:")
print(f"  Training Index:      {heaviest_idx}")
print(f"  True Identity:       Class {heaviest_true_label}")
print(f"  Final Sample Weight: {heaviest_weight:.6f} (baseline was {1.0/N_train:.6f}, {heaviest_weight / (1.0/N_train):.1f}x higher!)")
print(f"  Misclassified:       {mistakes_count} out of 100 rounds ({mistakes_count}% of weak learners got it wrong)")

# Let's inspect the top-3 predicted classes for this sample
unique_preds, pred_counts = np.unique(preds_history, return_counts=True)
sorted_pred_idx = np.argsort(pred_counts)[::-1]
print("  Weak learner prediction breakdown for this sample:")
for i in range(min(5, len(sorted_pred_idx))):
    p_class = unique_preds[sorted_pred_idx[i]]
    p_cnt = pred_counts[sorted_pred_idx[i]]
    print(f"    Class {p_class}: {p_cnt} times ({p_cnt}%)")

# Plot the heaviest face and its weight trajectory
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
axes[0].imshow(X_train[heaviest_idx].reshape(64, 64), cmap='gray')
axes[0].set_title(f"Heaviest Face (Idx {heaviest_idx}, Class {heaviest_true_label})\nMisclassified in {mistakes_count}/100 Rounds", fontsize=11)
axes[0].axis('off')

axes[1].plot(np.arange(1, 101), weight_trajectory, color='#d62728', lw=2)
axes[1].axhline(y=1.0/N_train, color='gray', linestyle=':', label=f'Initial Weight ({1.0/N_train:.4f})')
axes[1].set_xlabel('Boosting Round $t$', fontsize=11)
axes[1].set_ylabel('Sample Weight $w_i^{(t)}$', fontsize=11)
axes[1].set_title(f'Weight Evolution of Sample {heaviest_idx}', fontsize=12, fontweight='bold')
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend()

plt.tight_layout()
plt.savefig('q2_c_heaviest_face.png', dpi=300)
plt.close()
print("Saved q2_c_heaviest_face.png")

# ==============================================================================
# 2(d) Comparison of Baseline, Bagging, and Boosting on Clean and Masked Sets
# ==============================================================================
print("\n--- Question 2(d): Clean vs Masked Test Set Comparison ---")

# 1. Single Tree Baseline
single_tree = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, random_state=SEED)
single_tree.fit(X_train, y_train)

st_clean_acc = np.mean(single_tree.predict(X_test) == y_test)
st_clean_conf = np.mean(np.max(single_tree.predict_proba(X_test), axis=1))

st_masked_acc = np.mean(single_tree.predict(X_test_masked) == y_test)
st_masked_conf = np.mean(np.max(single_tree.predict_proba(X_test_masked), axis=1))

# 2. Final Bagging Model (B=100)
rng_b = np.random.default_rng(SEED)
bag_trees = []
tree_seeds = rng_b.integers(0, 1_000_000, size=100)
for b in range(100):
    boot = rng_b.choice(N_train, size=N_train, replace=True)
    t = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, random_state=int(tree_seeds[b]))
    t.fit(X_train[boot], y_train[boot])
    bag_trees.append(t)

bag_clean_preds = np.array([t.predict(X_test) for t in bag_trees])
bag_masked_preds = np.array([t.predict(X_test_masked) for t in bag_trees])

bag_clean_maj = np.zeros(len(y_test), dtype=int)
bag_clean_conf_list = np.zeros(len(y_test))
bag_masked_maj = np.zeros(len(y_test), dtype=int)
bag_masked_conf_list = np.zeros(len(y_test))

for i in range(len(y_test)):
    v_clean = np.bincount(bag_clean_preds[:, i], minlength=K)
    bag_clean_maj[i] = np.argmax(v_clean)
    bag_clean_conf_list[i] = np.max(v_clean) / 100.0
    
    v_masked = np.bincount(bag_masked_preds[:, i], minlength=K)
    bag_masked_maj[i] = np.argmax(v_masked)
    bag_masked_conf_list[i] = np.max(v_masked) / 100.0

bag_clean_acc = np.mean(bag_clean_maj == y_test)
bag_clean_conf = np.mean(bag_clean_conf_list)

bag_masked_acc = np.mean(bag_masked_maj == y_test)
bag_masked_conf = np.mean(bag_masked_conf_list)

# 3. Selected Boosting Model (Depth 4, 100 rounds)
boost_clean_pred = boost_model.predict(X_test)
boost_clean_acc = np.mean(boost_clean_pred == y_test)
boost_clean_conf = np.mean(boost_model.predict_confidence(X_test))

boost_masked_pred = boost_model.predict(X_test_masked)
boost_masked_acc = np.mean(boost_masked_pred == y_test)
boost_masked_conf = np.mean(boost_model.predict_confidence(X_test_masked))

print(f"{'Model':<24} | {'Clean Acc':<12} | {'Clean Conf':<12} | {'Masked Acc':<12} | {'Masked Conf':<12}")
print("-" * 80)
print(f"{'Single Tree Baseline':<24} | {st_clean_acc:<12.4f} | {st_clean_conf:<12.4f} | {st_masked_acc:<12.4f} | {st_masked_conf:<12.4f}")
print(f"{'Bagging (B=100)':<24} | {bag_clean_acc:<12.4f} | {bag_clean_conf:<12.4f} | {bag_masked_acc:<12.4f} | {bag_masked_conf:<12.4f}")
print(f"{'Boosting (Depth 4, 100 R)':<24} | {boost_clean_acc:<12.4f} | {boost_clean_conf:<12.4f} | {boost_masked_acc:<12.4f} | {boost_masked_conf:<12.4f}")

# Find test faces where Bagging and Boosting respond differently to the missing region:
# e.g., both correct on clean, but on masked bagging stays correct and boosting fails
diff_response_idx = []
for i in range(len(y_test)):
    bag_clean_correct = (bag_clean_maj[i] == y_test[i])
    bag_masked_correct = (bag_masked_maj[i] == y_test[i])
    boost_clean_correct = (boost_clean_pred[i] == y_test[i])
    boost_masked_correct = (boost_masked_pred[i] == y_test[i])
    
    # Bagging survives masking, boosting breaks
    if bag_clean_correct and bag_masked_correct and (not boost_masked_correct):
        diff_response_idx.append(i)

print(f"\nFound {len(diff_response_idx)} faces where Bagging survives masking (correct on both clean and masked), but Boosting fails on masked:")
print(f"Indices: {diff_response_idx}")

# Inspect the first such face
diff_face = diff_response_idx[0]
print(f"\nSelected Face for In-Depth Analysis: Test Sample Index {diff_face}")
print(f"  True Label:                   Class {y_test[diff_face]}")
print(f"  Bagging Clean Prediction:     Class {bag_clean_maj[diff_face]} (Confidence: {bag_clean_conf_list[diff_face]:.2f})")
print(f"  Bagging Masked Prediction:    Class {bag_masked_maj[diff_face]} (Confidence: {bag_masked_conf_list[diff_face]:.2f}) -> CORRECT")
print(f"  Boosting Clean Prediction:    Class {boost_clean_pred[diff_face]} (Confidence: {boost_model.predict_confidence(X_test)[diff_face]:.2f})")
print(f"  Boosting Masked Prediction:   Class {boost_masked_pred[diff_face]} (Confidence: {boost_model.predict_confidence(X_test_masked)[diff_face]:.2f}) -> WRONG")

# Visual plot of this face (Clean vs Masked) and model responses
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
axes[0].imshow(X_test[diff_face].reshape(64, 64), cmap='gray')
axes[0].set_title(f"Clean Face (Idx {diff_face}, True Class {y_test[diff_face]})\nBagging: C{bag_clean_maj[diff_face]} | Boosting: C{boost_clean_pred[diff_face]}", fontsize=10)
axes[0].axis('off')

axes[1].imshow(X_test_masked[diff_face].reshape(64, 64), cmap='gray')
axes[1].set_title(f"Masked Face (16x16 Center Masked)\nBagging: C{bag_masked_maj[diff_face]} (Correct!) | Boosting: C{boost_masked_pred[diff_face]} (Wrong!)", fontsize=10)
axes[1].axis('off')

plt.suptitle('Question 2(d): Differential Response to Center Occlusion', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('q2_d_robustness_face.png', dpi=300)
plt.close()
print("Saved q2_d_robustness_face.png")
