import numpy as np
from sklearn.tree import DecisionTreeClassifier

split_data = np.load("data/split_data.npz")
X_train_orig = np.ascontiguousarray(split_data['X_train'], dtype=np.float32)
y_train_orig = np.array(split_data['y_train'])
X_test = np.ascontiguousarray(split_data['X_test'], dtype=np.float32)
y_test = np.array(split_data['y_test'])

SEED = 26002
rng_resample = np.random.default_rng(SEED)
resampled_sets = []
for r in range(30):
    idx = []
    for c in range(40):
        c_orig = np.arange(6 * c, 6 * c + 6)
        c_resamp = rng_resample.choice(c_orig, size=6, replace=True)
        idx.extend(c_resamp)
    resampled_sets.append((X_train_orig[np.array(idx)], y_train_orig[np.array(idx)]))

# Evaluate depth 2 and 8
for d in [2, 8]:
    st_accs = []
    bag_accs = []
    for r in range(30):
        X_tr, y_tr = resampled_sets[r]
        # ST
        clf = DecisionTreeClassifier(max_depth=d, min_samples_leaf=1, random_state=SEED+r)
        clf.fit(X_tr, y_tr)
        st_accs.append(np.mean(clf.predict(X_test) == y_test))
        
        # Bagging 100 trees
        b_preds = []
        rng_b = np.random.default_rng(SEED + r)
        for b in range(100):
            boot = rng_b.choice(len(X_tr), len(X_tr), replace=True)
            t = DecisionTreeClassifier(max_depth=d, min_samples_leaf=1, random_state=int(rng_b.integers(0, 1000000)))
            t.fit(X_tr[boot], y_tr[boot])
            b_preds.append(t.predict(X_test))
        b_preds = np.array(b_preds)
        maj = np.array([np.argmax(np.bincount(b_preds[:, i], minlength=40)) for i in range(len(y_test))])
        bag_accs.append(np.mean(maj == y_test))
        
    st_m, bag_m = np.mean(st_accs), np.mean(bag_accs)
    print(f"Depth {d}: Single Tree = {st_m*100:.2f}%, Bagging = {bag_m*100:.2f}%, Improvement = {(bag_m - st_m)*100:+.2f}%")
