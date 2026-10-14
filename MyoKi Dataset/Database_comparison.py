import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, ttest_rel, ttest_ind, wilcoxon, mannwhitneyu

# Set font to Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

# Load the Excel file
file_path = "path_to/Database_comparison.xlsx"
df = pd.read_excel(file_path)

# Define column groups
colnames = df.columns.tolist()
# 0: NinaproDB7: EMG+IMU (independent)
# 1: NinaproDB7replica: EMG+IMU (same pop as 2,3)
# 2: NinaproDB7replica: FMG+IMU (same pop as 1,3)
# 3: NinaproDB7replica: EMG+FMG+IMU (same pop as 1,2)
# 4: MyoKi subset 1: EMG+IMU (independent)
# 5: MyoKi subset 2: EMG+FMG+IMU (independent)

# Prepare pairs for significance testing
pairs = [
    (colnames[1], colnames[2]),  # paired
    (colnames[1], colnames[3]),  # paired
    (colnames[2], colnames[3]),  # paired
    (colnames[4], colnames[5]),  # independent
    (colnames[0], colnames[1]),  # independent
    (colnames[0], colnames[2]),  # independent
    (colnames[0], colnames[3]),  # independent
]

results = []

for a, b in pairs:
    data_a = df[a].dropna()
    data_b = df[b].dropna()
    # Check if paired or independent
    paired = (a in colnames[1:4]) and (b in colnames[1:4])
    # Normality test
    if paired:
        diff = data_a.values - data_b.values
        _, p_norm = shapiro(diff)
        normal = p_norm > 0.05
        if normal:
            stat, p = ttest_rel(data_a, data_b)
            test_used = "Paired t-test"
        else:
            stat, p = wilcoxon(data_a, data_b)
            test_used = "Wilcoxon signed-rank"
    else:
        _, p_a = shapiro(data_a)
        _, p_b = shapiro(data_b)
        normal = (p_a > 0.05) and (p_b > 0.05)
        if normal:
            stat, p = ttest_ind(data_a, data_b, equal_var=False)
            test_used = "Independent t-test"
        else:
            stat, p = mannwhitneyu(data_a, data_b, alternative='two-sided')
            test_used = "Mann-Whitney U"
    results.append((a, b, test_used, p))
    print(f"{a} vs {b}: {test_used}, p = {p:.4g} (normal: {normal})")

# Create the box plot
plt.figure(figsize=(8, 4))
ax = sns.boxplot(
    data=df,
    showmeans=True,
    meanprops={"marker": "o", "markerfacecolor": "#4682B4", "markeredgecolor": "black"},
    boxprops={"facecolor": (0/255, 204/255, 0/255, 1)},
    whiskerprops={"color": "black"},
    capprops={"color": "black"},
    medianprops={"color": "black"}
)
plt.grid(axis="y", color="lightgray", linestyle="-", linewidth=0.7)
plt.xticks(rotation=15, ha="right")
plt.ylabel("R² Score")

# Annotate significant pairs on the plot
def add_significance(ax, pairs, data, colnames, y_offset=0.05, base_offset=-0.0):
    # Find the maximum y-value among all data points (not just box tops)
    ymax = max([max(df[c].dropna()) for c in colnames])
    ylim = ax.get_ylim()
    h = y_offset * (ylim[1] - ylim[0])
    base = ymax + base_offset * (ylim[1] - ylim[0])
    idx = 0
    for a, b, _, p in pairs:
        if p >= 0.05:
            continue  # Only show significant
        x1, x2 = colnames.index(a), colnames.index(b)
        y = base + h * idx
        ax.plot([x1, x1, x2, x2], [y, y+h/3, y+h/3, y], lw=0.8, c='k')
        # Set the significance star(s)
        if p < 0.001:
            text = '***'
        elif p < 0.01:
            text = '**'
        else:
            text = '*'
        ax.text((x1 + x2) / 2, y + -h/20, text, ha='center', va='bottom', color='k', fontsize=10)
        idx += 1
    # Optionally, extend the y-limits to make sure all annotations fit
    ax.set_ylim(ylim[0], base + h * (idx + 2))

# Filter only significant pairs
significant_pairs = [(a, b, t, p) for (a, b, t, p) in results if p < 0.05]
if significant_pairs:
    add_significance(ax, significant_pairs, df, colnames)

plt.ylim(top=1.09)
plt.tight_layout()
plt.show()
