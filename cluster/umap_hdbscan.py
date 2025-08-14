import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan

# === Config ===
processed_path = Path(r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\processed\PT_10_Day1")
output_path = processed_path / "cluster_outputs"
output_path.mkdir(exist_ok=True)

Trials = [
    "Coins", "Eggs_First", "Eggs_Second", "Grasp", "GTK", "Jenga", "Jenga_Standing",
    "Pegs", "Pegs_Standing", "SFO", "Tapes", "Tapes_Standing"
]
Trial_nos = [1, 2, 3, 4, 5]

# === Containers for combined data ===
all_features = []
all_times = []
all_frames = []
all_trial_labels = []
all_kinematics = []

EXPECTED_FEATURE_DIM = None  # Will be set after first valid file

for trial in Trials:
    for trial_no in Trial_nos:
        trial_id = f"{trial}_{trial_no}"
        npy_file = processed_path / f"{trial_id}_features.npy"
        starts_file = processed_path / f"{trial_id}_features.starts.npy"
        csv_file = processed_path / f"{trial_id}_features.csv"

        if not (npy_file.exists() and starts_file.exists() and csv_file.exists()):
            print(f"⚠️ Skipping {trial_id} (missing .npy, .starts.npy or .csv)")
            continue

        # Load sliding window features and window start frames
        X = np.load(npy_file)
        window_starts = np.load(starts_file)
        df = pd.read_csv(csv_file)

        # Basic checks
        if not {'time', 'frame'}.issubset(df.columns):
            print(f"❌ CSV {csv_file.name} missing 'time' or 'frame' columns. Skipping.")
            continue

        if len(window_starts) != X.shape[0]:
            print(f"❌ Mismatch in window_starts length and feature count for {trial_id}. Skipping.")
            continue

        if EXPECTED_FEATURE_DIM is None:
            EXPECTED_FEATURE_DIM = X.shape[1]
        elif X.shape[1] != EXPECTED_FEATURE_DIM:
            print(f"❌ Skipping {trial_id} due to shape mismatch: {X.shape[1]} vs expected {EXPECTED_FEATURE_DIM}")
            continue

        # Map window start frames to corresponding time and frames
        times_for_windows = []
        frames_for_windows = []
        for frame_start in window_starts:
            matched_row = df[df['frame'] == frame_start]
            if len(matched_row) == 1:
                times_for_windows.append(matched_row['time'].values[0])
                frames_for_windows.append(frame_start)
            else:
                # If no exact match, fallback to NaN or interpolate here if needed
                times_for_windows.append(np.nan)
                frames_for_windows.append(frame_start)

        # Extract kinematic features for each window start frame
        kinematics_for_windows = df[df['frame'].isin(window_starts)].reset_index(drop=True)
        # Handle mismatches by filling NaNs if needed
        if kinematics_for_windows.shape[0] != len(window_starts):
            print(f"⚠️ Kinematics frame count mismatch for {trial_id}, filling with NaNs")
            nan_df = pd.DataFrame(np.nan, index=range(len(window_starts)), columns=df.columns.drop(['time', 'frame']))
            kinematics_for_windows = nan_df

        # Extend global lists
        all_features.append(X)
        all_times.append(np.array(times_for_windows))
        all_frames.append(np.array(frames_for_windows))
        all_trial_labels.extend([trial_id] * len(window_starts))
        all_kinematics.append(kinematics_for_windows)

print(f"✅ Loaded data from {len(all_features)} trials")

if len(all_features) == 0:
    raise RuntimeError("❌ No valid data found to cluster!")

# Combine all
combined_features = np.vstack(all_features)
combined_times = np.concatenate(all_times)
combined_frames = np.concatenate(all_frames)
kinematics_df = pd.concat(all_kinematics, ignore_index=True)

print(f"✅ Combined data shape: {combined_features.shape}")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(combined_features)

# UMAP embedding
umap_model = umap.UMAP(n_neighbors=100, min_dist=0.5, n_components=2, metric='euclidean', random_state=42)
X_umap = umap_model.fit_transform(X_scaled)
print(f"✅ UMAP embedding shape: {X_umap.shape}")

# HDBSCAN clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=5)
labels = clusterer.fit_predict(X_umap)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"✅ Found {n_clusters} clusters, with {np.sum(labels==-1)} noise points")

# Build final DataFrame with all info
result_df = pd.DataFrame({
    "trial": all_trial_labels,
    "time": combined_times,
    "frame": combined_frames,
    "UMAP_1": X_umap[:, 0],
    "UMAP_2": X_umap[:, 1],
    "cluster": labels
})

# Add kinematic features aligned to windows
result_df = pd.concat([result_df.reset_index(drop=True), kinematics_df.reset_index(drop=True)], axis=1)

# Save output
output_csv = output_path / "PT_10_Day1_cluster_output.csv"
result_df.to_csv(output_csv, index=False)
print(f"✅ Saved clustering results to: {output_csv}")
