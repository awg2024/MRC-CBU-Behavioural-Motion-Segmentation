import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Config ---

cluster_path = r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\processed\PT_10_Day2\cluster_outputs\cluster_output.csv"
save_path = r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\processed_figures"

# Ensure save directory exists
os.makedirs(save_path, exist_ok=True)

# --- Load Data ---
cluster_data = pd.read_csv(cluster_path)

# Check for 'norm_time'; create if missing
if 'norm_time' not in cluster_data.columns:
    cluster_data['norm_time'] = cluster_data.groupby('trial')['time'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )

# --- Plot 1: Overall UMAP Projection (UMAP_1 vs UMAP_2) ---
sns.set(style="whitegrid", rc={"figure.figsize": (10, 6)})
plt.figure()
sns.scatterplot(
    x="UMAP_1", y="UMAP_2",
    hue="cluster", data=cluster_data,
    palette="tab10", s=5, linewidth=0
)
plt.title("UMAP Projection of All Trials")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()

umap_fig_path = os.path.join(save_path, "umap_projection.png")
plt.savefig(umap_fig_path, dpi=300)
print(f"UMAP projection saved to: {umap_fig_path}")
plt.show()
