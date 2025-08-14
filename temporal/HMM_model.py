import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from hmmlearn import hmm
import joblib

# === Load Data ===
input_csv = Path(r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\processed\PT_10_Day1\cluster_outputs\PT_10_Day1_cluster_output.csv")
df = pd.read_csv(input_csv)

# === Prepare Features ===

# One-hot encode cluster labels
cluster_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
cluster_1hot = cluster_encoder.fit_transform(df[['cluster']])

# Extract kinematic features
exclude_cols = {'trial', 'time', 'frame', 'UMAP_1', 'UMAP_2', 'cluster'}
kinematic_cols = [col for col in df.columns if col not in exclude_cols]
kinematic_features = df[kinematic_cols]

# Standardize kinematic features
kinematic_scaler = StandardScaler()
kinematic_scaled = kinematic_scaler.fit_transform(kinematic_features)

# Combine features for HMM
hmm_input = np.hstack([kinematic_scaled, cluster_1hot])

# === Sanity Check ===
assert not np.isnan(hmm_input).any(), "HMM input contains NaNs!"
print(f"HMM input shape: {hmm_input.shape}")

# === Helper: Compute BIC ===
def compute_bic(model, X):
    n_components = model.n_components
    n_features = X.shape[1]

    # BIC = -2 * log_likelihood + p * log(N)
    # p = number of parameters

    # Transition matrix: n_components * (n_components - 1) (excluding one row for sum=1 constraint)
    trans_params = n_components * (n_components - 1)

    # Start probabilities: n_components - 1 (due to sum-to-1 constraint)
    start_params = n_components - 1

    # Means and variances (diagonal covariances)
    mean_params = n_components * n_features
    cov_params = n_components * n_features  # diagonal cov

    n_params = trans_params + start_params + mean_params + cov_params

    log_likelihood = model.score(X)
    bic = -2 * log_likelihood + n_params * np.log(X.shape[0])
    return bic

# === Train HMM with Model Selection ===
min_states = 2
max_states = 16
best_model = None
lowest_bic = np.inf
bic_scores = []

print("\n🔍 Starting model selection via BIC...")
for n in range(min_states, max_states + 1):
    model = hmm.GaussianHMM(
        n_components=n,
        covariance_type="diag",
        n_iter=200,
        verbose=False,
        random_state=42
    )
    try:
        model.fit(hmm_input)
        bic = compute_bic(model, hmm_input)
        bic_scores.append((n, bic))
        print(f"   {n} states -> BIC: {bic:.2f}")
        if bic < lowest_bic:
            best_model = model
            lowest_bic = bic
            best_n = n
    except Exception as e:
        print(f"Failed for {n} states: {e}")

print(f"\n Best model: {best_n} hidden states (lowest BIC: {lowest_bic:.2f})")

# === Predict States and Save Outputs ===
df["HMM_state"] = best_model.predict(hmm_input)

# Paths for saving
model_path = input_csv.with_name("hmm_model.pkl")
scaler_path = input_csv.with_name("kinematic_scaler.pkl")
encoder_path = input_csv.with_name("cluster_encoder.pkl")
output_csv_path = input_csv.with_name("PT_14_Day4_cluster_output_with_hmm.csv")

# Save model and components
joblib.dump(best_model, model_path)
joblib.dump(kinematic_scaler, scaler_path)
joblib.dump(cluster_encoder, encoder_path)

# Save updated CSV
df.to_csv(output_csv_path, index=False)

# Output Summary
print(f"\n✅ Model saved to: {model_path}")
print(f"✅ Scaler saved to: {scaler_path}")
print(f"✅ Encoder saved to: {encoder_path}")
print(f"✅ Updated CSV saved to: {output_csv_path}")
print(df[["cluster", "HMM_state"]].head())
