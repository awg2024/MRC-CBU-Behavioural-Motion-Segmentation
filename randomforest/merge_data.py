from pathlib import Path
import pandas as pd
import re  # for extracting trial number from filename

# -------------------------------
# 📂 Paths
# -------------------------------
main_data_path = Path(r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\processed\PT_10_Day2\script_outputs\cluster_output_with_hmm.csv")
labelled_behaviours_paths = [
    Path(r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\processed\PT_10_Day2\script_outputs\Jenga_1-labelled-output.csv"), 
    Path(r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\processed\PT_10_Day2\script_outputs\Jenga_2-labelled-output.csv"), 
    Path(r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\processed\PT_10_Day2\script_outputs\Jenga_3-labelled-output.csv"), 
    Path(r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\processed\PT_10_Day2\script_outputs\Jenga_4-labelled-output.csv"), 
    Path(r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\processed\PT_10_Day2\script_outputs\Jenga_5-labelled-output.csv"), 
]
output_path = Path(r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\processed\PT_10_Day2\script_outputs\merged_data.csv")

# -------------------------------
# Load main dataset
# -------------------------------
df = pd.read_csv(main_data_path)
print(f"📄 Main data loaded: {df.shape[0]} rows")

# -------------------------------
# Load and concatenate labeled datasets
# -------------------------------
label_dfs = []
for path in labelled_behaviours_paths:
    tmp = pd.read_csv(path)
    
    # Extract trial number from filename using regex
    match = re.search(r'Jenga_(\d+)', path.name)
    if not match:
        raise ValueError(f"⚠️ Cannot extract trial number from filename {path.name}")
    trial_num = int(match.group(1))
    
    # Add trial column to tmp dataframe
    tmp['trial'] = trial_num
    
    # Check that 'frame' and 'label' exist
    if 'frame' not in tmp.columns or 'label' not in tmp.columns:
        raise ValueError(f"⚠️ File {path.name} must contain 'frame' and 'label' columns.")
    
    label_dfs.append(tmp[['trial', 'frame', 'label']])
    print(f"📄 Loaded {path.name}: {tmp.shape[0]} rows with trial={trial_num}")

all_labels = pd.concat(label_dfs, ignore_index=True)
print(f"🔗 Total combined labels: {all_labels.shape[0]}")

# -------------------------------
# Ensure trial columns have the same type for merging
# -------------------------------
df['trial'] = df['trial'].astype(str).str.extract(r'(\d+)').astype(int)
df['trial'] = df['trial'].astype(int)
all_labels['trial'] = all_labels['trial'].astype(int)

# -------------------------------
# Merge on trial + frame
# -------------------------------
if 'trial' in df.columns and 'frame' in df.columns:
    merged_df = pd.merge(df, all_labels, on=['trial', 'frame'], how='left')
else:
    raise ValueError("❌ Main data must contain both 'trial' and 'frame' columns.")

# -------------------------------
# Confirm label coverage
# -------------------------------
num_labeled = merged_df['label'].notna().sum()
print(f"✅ Labeled rows after merge: {num_labeled}")
print(f"🧮 Total rows in merged data: {merged_df.shape[0]}")

# -------------------------------
# Save to new CSV
# -------------------------------
merged_df.to_csv(output_path, index=False)
print(f"✅ Merged dataset saved to:\n{output_path}")
