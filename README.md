# Machine Learning: Assignment 2
## Bagging, Boosting and Bias-Variance on Face Recognition

**Student Roll Number:** `m26cse002`  
**Extracted Random Seed:** `26002` (`int("".join([c for c in "m26cse002" if c.isdigit()]))`)  
**Dataset:** Olivetti Faces (`sklearn.datasets.fetch_olivetti_faces`)  
- 40 identities, 10 images each ($64 \times 64 = 4096$ raw flattened pixels).
- **Stratified Split:** 6 train, 2 validation, 2 clean test per identity (240 train, 80 val, 80 test).
- **Robustness Test Set:** Centered $16 \times 16$ block (rows 24–39, cols 24–39) set to 0.0.

---

## Directory Structure

```text
Ml assignment 2/
├── data/
│   └── split_data.npz                  # Saved 60/20/20 stratified split & masked test set
├── figures/                            # High-resolution generated plots (300 DPI)
│   ├── q1_c_metrics_plot.png           # Q1(c): OOB acc, disagreement & vote margin vs B
│   ├── q1_d_error_cases.png            # Q1(d): Corrected vs both-wrong face case studies
│   ├── q2_b_diagnostics_plot.png       # Q2(b): SAMME 4-panel diagnostic curves over rounds
│   ├── q2_c_heaviest_face.png          # Q2(c): Heaviest training face & weight evolution
│   └── q2_d_robustness_face.png        # Q2(d): Clean vs masked differential response face
├── logs/                               # Benchmark logs and intermediate search JSONs
│   ├── q1_metrics.json
│   └── q2_grid_results.json
├── exploratory_scripts/                # Intermediate tuning and verification scripts
│   ├── prepare_data.py
│   ├── run_q1_a.py
│   ├── run_q1_b_c.py
│   ├── run_q1_d.py
│   ├── run_q2_all.py
│   ├── test_q2_a.py
│   └── test_q2_b_grid.py
├── solution_q1.py                      # Standalone, end-to-end solution for Question 1
├── solution_q2.py                      # Standalone, end-to-end solution for Question 2
├── requirements.txt                    # Python package dependencies
└── README.md                           # Documentation and reproduction guide
```

---

## Environment Setup

Install dependencies using Python 3.10+ (tested on Python 3.12):
```bash
pip install -r requirements.txt
```

---

## Reproduction Commands

### 1. Run Complete Question 1 Solution
```bash
python solution_q1.py
```
- Fits single tree baseline across 15 hyperparameter combinations.
- Freezes best model on clean validation accuracy (`max_depth=None, min_samples_leaf=1`).
- Implements Bagging from scratch for $B \in \{1, 5, 10, 25, 50, 100, 200\}$.
- Evaluates OOB accuracy, pairwise tree disagreement, and mean vote margin.
- Analyzes clean and masked test robustness, identifying corrected vs. uncorrectable error cases.
- Saves plots: `figures/q1_c_metrics_plot.png`, `figures/q1_d_error_cases.png`.

### 2. Run Complete Question 2 Solution
```bash
python solution_q2.py
```
- Logs complete walkthrough of Round 1 SAMME update on a 3-class toy dataset ($N=9$).
- Runs SAMME AdaBoost grid search over depths $\{1, 2, 3, 4\}$ and rounds $\{10, 25, 50, 100\}$.
- Selects `depth=4, rounds=100` via validation accuracy ($46.25\%$).
- Generates 4-panel diagnostics (weighted error, learner weight, max weight, top-10 weight).
- Compares Bagging vs. Boosting across 25, 50, and 100 learners (accuracies and train time).
- Inspects the heaviest training sample (Index 70, Class 11, misclassified in 88/100 rounds).
- Evaluates differential response to central occlusion (Index 57, Class 28).
- Saves plots: `figures/q2_b_diagnostics_plot.png`, `figures/q2_c_heaviest_face.png`, `figures/q2_d_robustness_face.png`.

### 3. Run Complete Question 3 Solution
```bash
python solution_q3.py
```
- Refits models across 30 resampled sets within each identity (6 faces from original 6).
- Computes measured $\text{Bias}^2$, $\text{Variance}$, and clean test accuracy for Single Tree, Bagging ($B=100$), and Boosting.
- Evaluates depth progression (Depth 2, Depth 8, Depth None) quantifying variance removal by Bagging.
- Explains why the largest accuracy gain occurs at Depth 8 (+29.25%) rather than Depth None (+28.42%).
- Outputs the comprehensive multi-model comparison table.
- Saves plot: `figures/q3_bias_variance_plot.png`.

---

## Key Results Summary

### Question 1 Summary
- **Single Tree Baseline:** Clean Test Acc = **$65.00\%$**, Masked Test Acc = **$23.75\%$**.
- **Final Bagged Model ($B=100$):** Clean Test Acc = **$88.75\%$**, Masked Test Acc = **$81.25\%$**.
- **Average Pairwise Disagreement:** Stable at $\approx \mathbf{76.2\%}$, providing error decorrelation.
- **Theoretical Bootstrap Expectation:** $151.90$ distinct samples ($63.29\%$), empirically matched at $151.69$ ($63.20\%$).

### Question 2 Summary
- **Selected Boosting Model:** Depth 4, 100 Rounds (Clean Val Acc = **$46.25\%$**, Clean Test Acc = **$37.50\%$**).
- **Bagging vs. Boosting:** Bagging achieves **$88.75\%$** clean test accuracy vs. Boosting's **$37.50\%$** due to severe weight concentration on difficult edge samples in boosting.
- **Occlusion Robustness:** Bagging retains **$81.25\%$** accuracy under center occlusion, whereas Single Tree drops to **$23.75\%$** and Boosting reaches **$35.00\%$**.

### Question 3 Summary
- **Single Tree Baseline:** $\text{Bias}^2 = 0.3634$, $\text{Variance} = 0.6157$, Test Acc = $51.04\% \pm 4.92\%$.
- **Final Bagged Model ($B=100$):** $\text{Bias}^2 = 0.4929$, $\text{Variance} = 0.0234$ (**$96.19\%$ variance reduction**), Test Acc = **$79.46\% \pm 3.63\%$**.
- **Selected Boosting Model:** $\text{Bias}^2 = 0.8878$ (severe underfitting), $\text{Variance} = 0.0170$, Test Acc = $29.88\% \pm 8.73\%$.
- **Largest Bagging Improvement:** Achieved at **Depth 8 (+29.25%)**, not at Depth None (+28.42%), due to high tree disagreement and moderate leaf occupancy.

