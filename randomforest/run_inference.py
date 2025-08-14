from pathlib import Path
import pandas as pd
import joblib
import numpy as np

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# 📂 Paths
# -------------------------------
full_data_path = Path(r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\processed\PT_08_Day1\script_outputs\merged_data.csv")
model_path = Path(r"C:\Users\ag05\Desktop\scripts\trial_segmentation\unsupervised\behavioural_classifier\models\jenga_behavioural_classifier.pkl")
output_path = Path(r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\processed\PT_08_Day1\script_outputs\third_iteration_prediction.csv")

# -------------------------------
# 🧠 Load model + feature list
# -------------------------------
if not model_path.exists():
    raise FileNotFoundError(f"❌ Trained model not found at: {model_path}")
clf, feature_cols = joblib.load(model_path)
print("✅ Loaded trained model and feature list.")

# -------------------------------
# 📊 Load full dataset
# -------------------------------
df = pd.read_csv(full_data_path)
print(f"📄 Full dataset loaded: {df.shape[0]} rows")

# -------------------------------
# 🧼 Ensure all feature columns exist
# -------------------------------
for col in feature_cols:
    if col not in df.columns:
        df[col] = 0  # fill missing features with 0

# -------------------------------
# 🔮 Predict labels for all rows
# -------------------------------
X_all = df[feature_cols]
df['predicted_label'] = clf.predict(X_all)
print(f"🔮 Predicted labels for all {df.shape[0]} rows.")

# -------------------------------
# ✅ Evaluate accuracy on labeled data
# -------------------------------
df_labeled = df[df['label'].notna()]
if not df_labeled.empty:
    y_true = df_labeled['label']
    y_pred = df_labeled['predicted_label']

    acc = accuracy_score(y_true, y_pred)
    print(f"📊 Accuracy on labeled data: {acc:.2%}\n")

    print("🔎 Classification Report:")
    print(classification_report(y_true, y_pred))

    # Confusion matrix heatmap
    cm = confusion_matrix(y_true, y_pred, labels=clf.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=clf.classes_, yticklabels=clf.classes_, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix: Original vs Predicted Labels")
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No labeled rows to evaluate accuracy.")

# -------------------------------
# 💾 Save prediction results
# -------------------------------
df.to_csv(output_path, index=False)
print(f"✅ Predictions saved to:\n{output_path}")
