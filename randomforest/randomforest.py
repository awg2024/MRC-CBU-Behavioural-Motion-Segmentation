from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# -------------------------------
# 📂 Paths
# -------------------------------
data_path = Path(r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\processed\PT_10_Day2\script_outputs\merged_data.csv")

model_path = Path(r"C:\Users\ag05\Desktop\scripts\trial_segmentation\unsupervised\behavioural_classifier\jenga_behavioural_classifier.pkl")

# -------------------------------
# 📊 Load labeled dataset
# -------------------------------
df = pd.read_csv(data_path)
print(f"📄 Labeled dataset loaded: {df.shape[0]} rows")

# -------------------------------
# 🧠 Feature matrix + labels
# -------------------------------
# Only exclude non-feature columns
non_feature_cols = ['label', 'frame', 'time', 'trial']
feature_cols = [col for col in df.columns if col not in non_feature_cols]

# Select numeric features only
X = df[feature_cols].select_dtypes(include=['number'])  # Filter to only usable features
y = df['label'].astype(str)

# Drop rows where any features or label are missing
df = df.dropna(subset=X.columns.tolist() + ['label'])

# Now recompute X and y from cleaned df
X = df[X.columns]
y = df['label'].astype(str)

print("📐 Feature shape:", X.shape)
print("🏷️ Label distribution:\n", y.value_counts())
print("🔍 Features being used:", X.columns.tolist())

# -------------------------------
# 🧠 Classifier setup
# -------------------------------
if model_path.exists():
    clf, feature_cols = joblib.load(model_path)
    print("✅ Loaded existing classifier.")
else:
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )
    feature_cols = X.columns.tolist()  # new model → take from current data
    print("⚠️ No classifier found — created a new one.")
# -------------------------------
# 🔀 Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
X, y, stratify=y, test_size=0.2, random_state=42
)

# -------------------------------
# 🧪 Train & Evaluate
# -------------------------------
clf.fit(X_train, y_train)
print("✅ Classifier trained.")


# Align test data: keep only the columns the model was trained on
X_test_aligned = X_test.copy()

# Drop any extra columns
X_test_aligned = X_test_aligned.reindex(columns=feature_cols, fill_value=0)

# Ensure it's a DataFrame with the exact same columns
X_test_aligned = pd.DataFrame(X_test_aligned, columns=feature_cols)

# Predict
y_pred = clf.predict(X_test_aligned)

print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Optional: Cross-validation
scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')
print("🔁 Cross-validated F1 scores:", scores)
print("📈 Mean F1:", scores.mean())

# -------------------------------
# 💾 Save model
# -------------------------------
joblib.dump((clf, feature_cols), model_path) # updated saving model here. 
print(f"✅ Model saved to: {model_path}")