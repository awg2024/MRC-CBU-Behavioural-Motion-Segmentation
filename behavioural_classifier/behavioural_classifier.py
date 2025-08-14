import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# === INPUT CSV with clusters + HMM states ===
csv_path = r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\processed\PT_14_Day4\cluster_outputs\PT_14_Day4_cluster_output_with_hmm.csv"

# === Load data ===
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV not found: {csv_path}")

df = pd.read_csv(csv_path)

# === Clean data ===
# Remove noise frames if cluster == -1
df = df[df['cluster'] != -1].dropna()

# === Labels for classifier ===
# For now, let's use cluster as behavior label — 
# you can replace this with your ground truth labels if available
y = df['cluster'].astype(str)  # Convert to string for classification

# === Feature selection ===
# Exclude columns that are not features or targets
exclude_cols = ['trial', 'time', 'frame', 'cluster', 'UMAP_1', 'UMAP_2', 'behavior_label']  # behavior_label may or may not exist
if 'behavior_label' in df.columns:
    exclude_cols.append('behavior_label')

# Include HMM_state explicitly as feature
features_cols = [col for col in df.columns if col not in exclude_cols]
if 'HMM_state' not in features_cols:
    features_cols.append('HMM_state')  # Just in case

X = df[features_cols]

print(f"✅ Using {X.shape[0]} samples and {X.shape[1]} features including HMM_state.")

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# === Train Random Forest ===
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# === Save model ===
model_path = r"C:\Users\ag05\Desktop\scripts\trial_segmentation\unsupervised\behavioural_classifier\behavioral_classifier.pkl"
joblib.dump(clf, model_path)
print(f"✅ Random Forest model saved to:\n{model_path}")
