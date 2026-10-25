import numpy as np
from sklearn.tree import DecisionTreeClassifier
import time

data = np.load("split_data.npz")
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

SEED = 26002
K = 40
N_train = len(X_train) # 240

depths = [1, 2, 3, 4]
rounds_list = [10, 25, 50, 100]
max_rounds = 100

class SAMME_Classifier:
    def __init__(self, base_depth, n_rounds, random_state=SEED):
        self.base_depth = base_depth
        self.n_rounds = n_rounds
        self.random_state = random_state
        self.learners = []
        self.alphas = []
        self.weighted_errors = []
        self.max_sample_weights = []
        self.top10_weights = []
        
    def fit(self, X, y, X_val=None, y_val=None):
        N = len(X)
        w = np.ones(N) / N
        rng = np.random.default_rng(self.random_state)
        random_error_threshold = (K - 1.0) / K
        
        val_acc_history = {}
        
        for t in range(self.n_rounds):
            tree_seed = int(rng.integers(0, 1_000_000))
            clf = DecisionTreeClassifier(max_depth=self.base_depth, random_state=tree_seed)
            clf.fit(X, y, sample_weight=w)
            
            preds = clf.predict(X)
            incorrect = (preds != y).astype(float)
            
            # Weighted error
            eps_t = np.sum(w * incorrect) / np.sum(w)
            
            # Handle edge cases
            if eps_t <= 1e-10:
                # Perfect learner
                alpha_t = np.log((1.0 - 1e-10) / 1e-10) + np.log(K - 1)
                self.learners.append(clf)
                self.alphas.append(alpha_t)
                self.weighted_errors.append(eps_t)
                self.max_sample_weights.append(np.max(w))
                self.top10_weights.append(np.sum(np.sort(w)[-10:]))
                print(f"Round {t+1}: Perfect classification achieved (eps=0).")
                break
                
            if eps_t >= random_error_threshold:
                print(f"Round {t+1}: Weak learner error {eps_t:.4f} >= {random_error_threshold:.4f} (no better than random). Stopping early.")
                break
                
            alpha_t = np.log((1.0 - eps_t) / eps_t) + np.log(K - 1)
            
            # Update weights
            w = w * np.exp(alpha_t * incorrect)
            w = w / np.sum(w)
            
            self.learners.append(clf)
            self.alphas.append(alpha_t)
            self.weighted_errors.append(eps_t)
            self.max_sample_weights.append(np.max(w))
            self.top10_weights.append(np.sum(np.sort(w)[-10:]))
            
            # Record val accuracy at target rounds
            round_num = t + 1
            if round_num in rounds_list and X_val is not None:
                val_acc = self.score_rounds(X_val, y_val, round_num)
                val_acc_history[round_num] = val_acc
                
        return val_acc_history

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

    def score_rounds(self, X, y, n_used):
        preds = self.predict(X, n_used)
        return np.mean(preds == y)

print("Running grid search over depths [1, 2, 3, 4] and rounds [10, 25, 50, 100]...")
all_grid_results = []

for d in depths:
    print(f"\n--- Depth {d} ---")
    start_time = time.time()
    model = SAMME_Classifier(base_depth=d, n_rounds=max_rounds, random_state=SEED)
    val_acc_map = model.fit(X_train, y_train, X_val, y_val)
    elapsed = time.time() - start_time
    print(f"Depth {d} trained {len(model.learners)} rounds in {elapsed:.2f}s")
    for r in rounds_list:
        if r <= len(model.learners):
            val_acc = val_acc_map.get(r, model.score_rounds(X_val, y_val, r))
            train_acc = model.score_rounds(X_train, y_train, r)
            test_acc = model.score_rounds(X_test, y_test, r)
            all_grid_results.append({
                'depth': d,
                'rounds': r,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'test_acc': test_acc
            })
            print(f"  Rounds {r:>3}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}, Test Acc = {test_acc:.4f}")

import json
with open("q2_grid_results.json", "w") as f:
    json.dump(all_grid_results, f, indent=4)
