import numpy as np
import time
import os
from sklearn.tree import DecisionTreeClassifier
from concurrent.futures import ProcessPoolExecutor
import json

# Ensure directories
os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Load split
split_data = np.load("data/split_data.npz")
X_train_orig = np.ascontiguousarray(split_data['X_train'], dtype=np.float32)
y_train_orig = np.array(split_data['y_train'])
X_test = np.ascontiguousarray(split_data['X_test'], dtype=np.float32)
y_test = np.array(split_data['y_test'])
X_test_masked = np.ascontiguousarray(split_data['X_test_masked'], dtype=np.float32)

SEED = 26002
K = 40
N_test = len(y_test)
R = 30

# One-hot target matrix
E = np.zeros((N_test, K), dtype=np.float32)
for i in range(N_test):
    E[i, y_test[i]] = 1.0

# 1. Generate 30 resampled training sets deterministically
rng_resample = np.random.default_rng(SEED)
resampled_sets = []
for r in range(R):
    idx = []
    for c in range(40):
        c_orig = np.arange(6 * c, 6 * c + 6)
        c_resamp = rng_resample.choice(c_orig, size=6, replace=True)
        idx.extend(c_resamp)
    idx = np.array(idx)
    resampled_sets.append((X_train_orig[idx], y_train_orig[idx]))

print(f"Generated {R} resampled training sets.")

def compute_bias_var(P_r, y_true):
    # P_r shape: (R, N_test, K)
    p_bar = np.mean(P_r, axis=0) # (N_test, K)
    bias_sq = float(np.mean(np.sum((p_bar - E)**2, axis=1)))
    var = float(np.mean(np.sum((P_r - p_bar)**2, axis=2)))
    return bias_sq, var

# ==============================================================================
# Helper worker functions for multiprocessing
# ==============================================================================
def fit_single_tree(args):
    r, depth, seed = args
    X_tr, y_tr = resampled_sets[r]
    clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=1, random_state=seed)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_test)
    acc = np.mean(preds == y_test)
    
    # Class probability vector / one-hot vote vector
    # To be consistent with majority voting vote fraction:
    P = np.zeros((N_test, K), dtype=np.float32)
    for i in range(N_test):
        P[i, preds[i]] = 1.0
        
    # Also evaluate on masked test
    preds_masked = clf.predict(X_test_masked)
    acc_masked = np.mean(preds_masked == y_test)
    
    return r, depth, P, acc, acc_masked

def fit_bagging(args):
    r, depth, B, seed = args
    X_tr, y_tr = resampled_sets[r]
    N_tr = len(y_tr)
    rng = np.random.default_rng(seed)
    
    tree_seeds = rng.integers(0, 1_000_000, size=B)
    all_preds = np.zeros((B, N_test), dtype=int)
    all_preds_masked = np.zeros((B, N_test), dtype=int)
    
    for b in range(B):
        boot = rng.choice(N_tr, size=N_tr, replace=True)
        clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=1, random_state=int(tree_seeds[b]))
        clf.fit(X_tr[boot], y_tr[boot])
        all_preds[b] = clf.predict(X_test)
        all_preds_masked[b] = clf.predict(X_test_masked)
        
    # Vote vector P (N_test, K)
    P = np.zeros((N_test, K), dtype=np.float32)
    ens_preds = np.zeros(N_test, dtype=int)
    ens_preds_masked = np.zeros(N_test, dtype=int)
    
    for i in range(N_test):
        v = np.bincount(all_preds[:, i], minlength=K)
        P[i] = v / B
        ens_preds[i] = np.argmax(v)
        ens_preds_masked[i] = np.argmax(np.bincount(all_preds_masked[:, i], minlength=K))
        
    acc = np.mean(ens_preds == y_test)
    acc_masked = np.mean(ens_preds_masked == y_test)
    return r, depth, P, acc, acc_masked

def fit_boosting(args):
    r, depth, rounds, seed = args
    X_tr, y_tr = resampled_sets[r]
    N_tr = len(y_tr)
    w = np.ones(N_tr, dtype=np.float64) / N_tr
    rng = np.random.default_rng(seed)
    random_error_threshold = (K - 1.0) / K
    
    learners = []
    alphas = []
    
    for t in range(rounds):
        t_seed = int(rng.integers(0, 1_000_000))
        clf = DecisionTreeClassifier(max_depth=depth, random_state=t_seed)
        clf.fit(X_tr, y_tr, sample_weight=w)
        
        preds_tr = clf.predict(X_tr)
        incorrect = (preds_tr != y_tr).astype(float)
        eps_t = np.sum(w * incorrect) / np.sum(w)
        
        if eps_t <= 1e-10:
            alpha_t = np.log((1.0 - 1e-10) / 1e-10) + np.log(K - 1)
            learners.append(clf)
            alphas.append(alpha_t)
            break
        if eps_t >= random_error_threshold:
            break
            
        alpha_t = np.log((1.0 - eps_t) / eps_t) + np.log(K - 1)
        learners.append(clf)
        alphas.append(alpha_t)
        
        w = w * np.exp(alpha_t * incorrect)
        w = w / np.sum(w)
        
    alphas = np.array(alphas)
    sum_alpha = np.sum(alphas)
    
    # Compute normalized weighted vote vector q_k(x) on X_test
    S = np.zeros((N_test, K), dtype=np.float32)
    S_masked = np.zeros((N_test, K), dtype=np.float32)
    for t in range(len(learners)):
        pt = learners[t].predict(X_test)
        pt_m = learners[t].predict(X_test_masked)
        for i in range(N_test):
            S[i, pt[i]] += alphas[t]
            S_masked[i, pt_m[i]] += alphas[t]
            
    P = S / sum_alpha
    ens_preds = np.argmax(S, axis=1)
    acc = np.mean(ens_preds == y_test)
    
    ens_preds_m = np.argmax(S_masked, axis=1)
    acc_masked = np.mean(ens_preds_m == y_test)
    
    return r, depth, P, acc, acc_masked

if __name__ == '__main__':
    t_start = time.time()
    n_workers = 10
    print(f"Using {n_workers} parallel worker processes...")

    # 1. Single Trees at depths: 2, 8, None
    print("\n--- Fitting Single Trees across 3 depths (2, 8, None) x 30 refits ---")
    st_tasks = []
    rng_seeds = np.random.default_rng(SEED)
    depth_list = [2, 8, None]
    for d in depth_list:
        for r in range(R):
            s = int(rng_seeds.integers(0, 1_000_000))
            st_tasks.append((r, d, s))
            
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        st_results = list(executor.map(fit_single_tree, st_tasks))
        
    # Organize single tree results by depth
    st_P = {d: np.zeros((R, N_test, K)) for d in depth_list}
    st_accs = {d: np.zeros(R) for d in depth_list}
    st_accs_m = {d: np.zeros(R) for d in depth_list}
    for r, d, P, acc, acc_m in st_results:
        st_P[d][r] = P
        st_accs[d][r] = acc
        st_accs_m[d][r] = acc_m

    # 2. Bagging 100 trees at depths: 2, 8, None
    print("--- Fitting 100-Tree Bagged Ensembles across 3 depths (2, 8, None) x 30 refits ---")
    bag_tasks = []
    for d in depth_list:
        for r in range(R):
            s = int(rng_seeds.integers(0, 1_000_000))
            bag_tasks.append((r, d, 100, s))
            
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        bag_results = list(executor.map(fit_bagging, bag_tasks))
        
    bag_P = {d: np.zeros((R, N_test, K)) for d in depth_list}
    bag_accs = {d: np.zeros(R) for d in depth_list}
    bag_accs_m = {d: np.zeros(R) for d in depth_list}
    for r, d, P, acc, acc_m in bag_results:
        bag_P[d][r] = P
        bag_accs[d][r] = acc
        bag_accs_m[d][r] = acc_m

    # 3. Boosting (Depth 4, 100 rounds) x 30 refits
    print("--- Fitting Boosting (Depth 4, 100 Rounds) x 30 refits ---")
    boost_tasks = []
    for r in range(R):
        s = int(rng_seeds.integers(0, 1_000_000))
        boost_tasks.append((r, 4, 100, s))
        
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        boost_results = list(executor.map(fit_boosting, boost_tasks))
        
    boost_P = np.zeros((R, N_test, K))
    boost_accs = np.zeros(R)
    boost_accs_m = np.zeros(R)
    for r, d, P, acc, acc_m in boost_results:
        boost_P[r] = P
        boost_accs[r] = acc
        boost_accs_m[r] = acc_m

    total_time = time.time() - t_start
    print(f"\nCompleted all 30 refits across all models in {total_time:.1f}s ({total_time/60:.2f} min)!\n")

    # ==============================================================================
    # Compute Metrics for Q3(a)
    # ==============================================================================
    print("=" * 80)
    print("QUESTION 3(a): Bias^2, Variance, and Test Accuracy across 30 Refits")
    print("=" * 80)
    
    # Selected Single Tree (None)
    st_b2, st_var = compute_bias_var(st_P[None], y_test)
    st_acc_mean, st_acc_std = np.mean(st_accs[None]), np.std(st_accs[None])
    
    # Selected Bagging (None, B=100)
    bag_b2, bag_var = compute_bias_var(bag_P[None], y_test)
    bag_acc_mean, bag_acc_std = np.mean(bag_accs[None]), np.std(bag_accs[None])
    
    # Selected Boosting (Depth 4, 100 rounds)
    boost_b2, boost_var = compute_bias_var(boost_P, y_test)
    boost_acc_mean, boost_acc_std = np.mean(boost_accs), np.std(boost_accs)
    
    print(f"{'Model':<25} | {'Bias^2':<12} | {'Variance':<12} | {'Total MSE':<12} | {'Test Acc (Mean ± Std)':<22}")
    print("-" * 88)
    print(f"{'Selected Single Tree':<25} | {st_b2:<12.4f} | {st_var:<12.4f} | {st_b2 + st_var:<12.4f} | {st_acc_mean*100:.2f}% ± {st_acc_std*100:.2f}%")
    print(f"{'Final Bagging (B=100)':<25} | {bag_b2:<12.4f} | {bag_var:<12.4f} | {bag_b2 + bag_var:<12.4f} | {bag_acc_mean*100:.2f}% ± {bag_acc_std*100:.2f}%")
    print(f"{'Selected Boosting (D4, 100R)':<25} | {boost_b2:<12.4f} | {boost_var:<12.4f} | {boost_b2 + boost_var:<12.4f} | {boost_acc_mean*100:.2f}% ± {boost_acc_std*100:.2f}%")

    # ==============================================================================
    # Compute Metrics for Q3(b)
    # ==============================================================================
    print("\n" + "=" * 80)
    print("QUESTION 3(b): Depth vs Bias, Variance, and Variance Removed by Bagging")
    print("=" * 80)
    print(f"{'Depth':<10} | {'ST Bias^2':<10} | {'ST Var':<10} | {'Bag Bias^2':<12} | {'Bag Var':<10} | {'Var Removed':<14} | {'% Var Reduction':<16}")
    print("-" * 88)
    
    depth_labels = {2: "2 (Shallow)", 8: "8 (Interm)", None: "None (Deep)"}
    for d in depth_list:
        sb2, svar = compute_bias_var(st_P[d], y_test)
        bb2, bvar = compute_bias_var(bag_P[d], y_test)
        var_removed = svar - bvar
        pct_reduced = (var_removed / svar) * 100.0 if svar > 0 else 0.0
        print(f"{depth_labels[d]:<10} | {sb2:<10.4f} | {svar:<10.4f} | {bb2:<12.4f} | {bvar:<10.4f} | {var_removed:<14.4f} | {pct_reduced:<16.2f}%")

    # ==============================================================================
    # Compute Metrics for Q3(d) Final Comparison Table
    # ==============================================================================
    print("\n" + "=" * 80)
    print("QUESTION 3(d): Final Comprehensive Model Comparison Table")
    print("=" * 80)
    print(f"{'Model':<25} | {'Clean Acc':<12} | {'Masked Acc':<12} | {'Bias^2':<12} | {'Variance':<12}")
    print("-" * 77)
    print(f"{'Single Tree (Selected)':<25} | {st_acc_mean:<12.4f} | {np.mean(st_accs_m[None]):<12.4f} | {st_b2:<12.4f} | {st_var:<12.4f}")
    print(f"{'Bagging (B=100, Selected)':<25} | {bag_acc_mean:<12.4f} | {np.mean(bag_accs_m[None]):<12.4f} | {bag_b2:<12.4f} | {bag_var:<12.4f}")
    print(f"{'Boosting (D4, 100R)':<25} | {boost_acc_mean:<12.4f} | {np.mean(boost_accs_m):<12.4f} | {boost_b2:<12.4f} | {boost_var:<12.4f}")

    # Save all results to logs/q3_results.json
    results_q3 = {
        'st_bias_sq': st_b2, 'st_var': st_var, 'st_acc_mean': st_acc_mean, 'st_acc_std': st_acc_std,
        'bag_bias_sq': bag_b2, 'bag_var': bag_var, 'bag_acc_mean': bag_acc_mean, 'bag_acc_std': bag_acc_std,
        'boost_bias_sq': boost_b2, 'boost_var': boost_var, 'boost_acc_mean': boost_acc_mean, 'boost_acc_std': boost_acc_std,
    }
    with open("logs/q3_results.json", "w") as f:
        json.dump(results_q3, f, indent=4)
    print("\nSaved Question 3 results to 'logs/q3_results.json'.")
