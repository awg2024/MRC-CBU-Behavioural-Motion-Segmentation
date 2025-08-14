"""

Here this is the predict function for the random forest classifier. Here i hope to 
feed the classifier data - clustered, hmm_states - then the classifier through the data it's been trained on
can associate and different this data into output - i want a defined set of behaviours? predicting behavioural labels 

"""

from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from pathlib import Path

# ---------- Setup ----------
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode
import numpy as np
import cv2
import pandas as pd
from pathlib import Path


# ---------- Paths ----------
model_path = Path(r"C:\Users\ag05\Desktop\scripts\trial_segmentation\unsupervised\behavioural_classifier\behavioural_classifier.pkl")
data_path = Path(r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\processed\PT_15_Day1\cluster_outputs\cluster_output_with_hmm.csv")
output_csv = data_path.with_name("cluster_output_with_predictions.csv")
confusion_plot_path = data_path.with_name("confusion_matrix.png")

# ---------- Load classifier and data ----------
clf = joblib.load(model_path)
print("✅ Classifier loaded.")

df = pd.read_csv(data_path)
print(f"📄 Data loaded: {df.shape[0]} rows")

# ---------- Feature prep ----------
non_feature_cols = ['HMM_state', 'behavior_label', 'cluster_enc', 'frame.1', 'time.1']
feature_cols = [col for col in df.columns if col not in non_feature_cols]
X_all = df[feature_cols].select_dtypes(include=['number'])

# ---------- Predict ----------
df['predicted_behavior'] = clf.predict(X_all)
print("✅ Behavior predictions complete.")

# ---------- Evaluate ----------
if 'behavior_label' in df.columns:
    labeled_df = df.dropna(subset=['behavior_label'])
    y_true = labeled_df['behavior_label']
    y_pred = labeled_df['predicted_behavior']

    print("\n📊 Classification Report:\n")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=clf.classes_)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=clf.classes_, yticklabels=clf.classes_, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(confusion_plot_path)
    print(f"📈 Confusion matrix saved to: {confusion_plot_path}")
else:
    print("⚠️ No ground truth for evaluation.")

# ---------- Save predictions ----------
df.to_csv(output_csv, index=False)
print(f"✅ Predictions saved to: {output_csv}")


### HMM BLOCK 

# ---------- HMM + Rolling Mode Smoothing Fix ----------
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode
import numpy as np

# Label encode predicted behaviors
le = LabelEncoder()
X_pred = le.fit_transform(df['predicted_behavior'])
X_pred_seq = X_pred.reshape(-1, 1)

# Fit HMM
n_behaviors = len(le.classes_)
hmm_model = hmm.MultinomialHMM(n_components=n_behaviors, n_iter=100)
hmm_model.fit(X_pred_seq)

# Predict HMM-smoothed sequence
logprob, hmm_states = hmm_model.decode(X_pred_seq, algorithm="viterbi")
df['hmm_behavior'] = le.inverse_transform(hmm_states)

# Convert to integers again for rolling mode
hmm_ints = le.transform(df['hmm_behavior'])

# Define rolling mode (numeric only)
def rolling_mode(data, window=7):
    padded = np.pad(data, (window//2, window//2), mode='edge')
    return np.array([mode(padded[i:i+window], keepdims=False).mode for i in range(len(data))])

smoothed_ints = rolling_mode(hmm_ints, window=7)
df['smoothed_behavior'] = le.inverse_transform(smoothed_ints)


# -------------------------------
# 📂 Paths
# -------------------------------
prediction_csv = Path(r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\processed\PT_15_Day1\cluster_outputs\cluster_output_with_predictions.csv")
video_path = Path(r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\unprocessed\PT_15_Day1\Grasp_5\all_3d\2001-01-01\videos-raw\Grasp_5-camB.mp4")
output_video = Path(r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\processed\PT_15_Day1\cluster_outputs\Grasp_5-camB-forest-prediction.mp4")
model_path = Path(r"C:\Users\ag05\Desktop\scripts\trial_segmentation\unsupervised\behavioural_classifier\behavioural_classifier.pkl")
# For probability estimates (RF only, not HMM)
proba_df = clf.predict_proba(X_all)
proba_labels = clf.classes_

# Build a color map for each behavior
import random
import matplotlib.colors as mcolors

behavior_list = list(df['smoothed_behavior'].unique())
color_map = {label: tuple(int(c*255) for c in mcolors.to_rgb(color))
             for label, color in zip(behavior_list, random.sample(list(mcolors.CSS4_COLORS.values()), len(behavior_list)))}


import cv2

# Load original video to extract properties
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise IOError(f"Could not open video: {video_path}")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()



cap = cv2.VideoCapture(str(video_path))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.6
thickness = 4
org = (30, 60)

print("🎞️ Creating overlay with smoothed predictions and probabilities...")

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_idx >= len(df):
        break

    label = df.loc[frame_idx, 'smoothed_behavior']
    label_index = list(proba_labels).index(df.loc[frame_idx, 'predicted_behavior'])
    prob = proba_df[frame_idx][label_index]
    label_text = f"{label} ({prob:.2f})"

    color = color_map.get(label, (0, 255, 0))  # default green
    cv2.putText(frame, label_text, org, font, font_scale, color, thickness, cv2.LINE_AA)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print(f"✅ Overlay video saved to: {output_video}")
