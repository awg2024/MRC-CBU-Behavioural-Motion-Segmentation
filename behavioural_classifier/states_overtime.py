import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Load your HMM-labeled DataFrame
# Replace with your actual path if not already loaded
df = pd.read_csv(r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\processed\PT_15_Day1\cluster_outputs\cluster_output_with_hmm.csv")

# Filter for the trial you’re interested in (if multiple trials are present)
trial_name = "Jenga_5"  # Adjust this
df = df[df['trial'] == trial_name].copy()

# Ensure time and HMM columns exist
assert 'time' in df.columns
assert 'HMM_state' in df.columns or 'HMM_state_smooth' in df.columns

# Optional smoothing (if not already done)
if 'HMM_state_smooth' not in df.columns:
    from scipy.stats import mode
    df['HMM_state_smooth'] = (
        df['HMM_state']
        .rolling(window=5, center=True)
        .apply(lambda x: mode(x, keepdims=True)[0][0])
    )

# === Plotting
plt.figure(figsize=(16, 4))
sns.set_style("whitegrid")

plt.plot(df['time'], df['HMM_state'], label='Raw HMM State', alpha=0.3, linewidth=1)
plt.plot(df['time'], df['HMM_state_smooth'], label='Smoothed HMM State', linewidth=2)

plt.title(f"HMM State Transitions Over Time – {trial_name}")
plt.xlabel("Time (s)")
plt.ylabel("HMM State")
plt.yticks(sorted(df['HMM_state'].unique()))
plt.legend()
plt.tight_layout()
plt.show()
