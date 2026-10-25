import numpy as np
import matplotlib.pyplot as plt

# Data from measured Question 3 calculations
depths = ['Depth 2\n(Shallow)', 'Depth 8\n(Intermediate)', 'Depth None\n(Deepest)']
st_bias = [1.8116, 1.3780, 0.3634]
st_var  = [0.0692, 0.2979, 0.6157]
bag_bias = [0.9296, 0.7987, 0.4929]
bag_var  = [0.0102, 0.0155, 0.0234]

st_acc = [6.04, 17.38, 51.04]
bag_acc = [13.37, 46.62, 79.46]

x = np.arange(len(depths))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1: Bias^2 and Variance across Depths
axes[0].bar(x - width/2, st_var, width, label='Single Tree Variance', color='#d62728', alpha=0.85)
axes[0].bar(x - width/2, st_bias, width, bottom=st_var, label=r'Single Tree Bias$^2$', color='#ff9896', alpha=0.85)

axes[0].bar(x + width/2, bag_var, width, label='Bagging (100 Trees) Variance', color='#1f77b4', alpha=0.85)
axes[0].bar(x + width/2, bag_bias, width, bottom=bag_var, label=r'Bagging Bias$^2$', color='#aec7e8', alpha=0.85)

axes[0].set_xticks(x)
axes[0].set_xticklabels(depths, fontsize=11)
axes[0].set_ylabel(r'Decomposition Value (Bias$^2$ + Variance)', fontsize=11)
axes[0].set_title(r'Question 3(b): Bias$^2$ and Variance vs. Tree Depth', fontsize=12, fontweight='bold')
axes[0].legend(loc='upper right', fontsize=9)
axes[0].grid(axis='y', linestyle='--', alpha=0.6)

# Plot 2: Test Accuracy & Improvement
axes[1].plot(x, st_acc, 'o--', label='Single Tree Accuracy (%)', color='#d62728', lw=2)
axes[1].plot(x, bag_acc, 's-', label='Bagging (100 Trees) Accuracy (%)', color='#1f77b4', lw=2)
improvement = np.array(bag_acc) - np.array(st_acc)
for i in range(len(x)):
    axes[1].annotate(f'+{improvement[i]:.2f}% gain', (x[i], bag_acc[i] + 2), 
                    ha='center', fontsize=10, fontweight='bold', color='#2ca02c')

axes[1].set_xticks(x)
axes[1].set_xticklabels(depths, fontsize=11)
axes[1].set_ylabel('Clean Test Accuracy (%)', fontsize=11)
axes[1].set_ylim(0, 95)
axes[1].set_title('Question 3(c): Test Accuracy & Bagging Improvement vs. Depth', fontsize=12, fontweight='bold')
axes[1].legend(loc='upper left', fontsize=10)
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('figures/q3_bias_variance_plot.png', dpi=300)
plt.savefig('q3_bias_variance_plot.png', dpi=300)
plt.close()
print("Saved figures/q3_bias_variance_plot.png")
