import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier

# Load split data
split_data = np.load("data/split_data.npz")
X_train_orig = split_data['X_train']
y_train_orig = split_data['y_train']
X_test = split_data['X_test']
y_test = split_data['y_test']
X_test_masked = split_data['X_test_masked']

SEED = 26002
rng = np.random.default_rng(SEED)

N_test = len(y_test) # 80
K = 40
R = 30

# One-hot target matrix E (N_test, K)
E = np.zeros((N_test, K))
for i in range(N_test):
    E[i, y_test[i]] = 1.0

# Pre-generate the 30 resampled training sets
# "sampling with replacement, within each identity, six faces from that identity's original six training faces"
resampled_train_sets = []
for r in range(R):
    r_idx = []
    for c in range(40):
        # original training indices for identity c are 6*c to 6*c+6
        id_range = np.arange(6 * c, 6 * c + 6)
        boot_c = rng.choice(id_range, size=6, replace=True)
        r_idx.extend(boot_c)
    r_idx = np.array(r_idx)
    resampled_train_sets.append((X_train_orig[r_idx], y_train_orig[r_idx]))

print(f"Generated {R} resampled training sets.")
print("Testing speed for 1 refit...")
t0 = time.time()
# Single tree fit
st = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, random_state=int(rng.integers(0, 1000000)))
st.fit(resampled_train_sets[0][0], resampled_train_sets[0][1])
st_time = time.time() - t0
print(f"Single tree fit time: {st_time:.4f}s")
