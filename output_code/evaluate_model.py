import cv2
import pandas as pd
import joblib
import os
from pathlib import Path
import numpy as np

# === Paths ===
model_path = Path(r"C:\Users\ag05\Desktop\scripts\trial_segmentation\unsupervised\behavioural_classifier\behavioural_classifier.pkl")
video_path = Path(r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\unprocessed\PT_12_Day3\Grasp_1\all_3d\2001-01-01\videos-raw\Grasp_1-camB.mp4")
output_video_path = Path(r"C:\Users\ag05\Desktop\scripts\trial_segmentation\unsupervised\randomforest\Grasp_1-camB-randomforest.mp4")
csv_path = Path(r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\processed\PT_12_Day3\cluster_outputs\cluster_output_with_hmm.csv")

# === Load classifier ===
if not model_path.exists():
    raise FileNotFoundError(f"❌ Could not find model at: {model_path}")
clf = joblib.load(model_path)
print("✅ Loaded model")

# === Load data ===
if not csv_path.exists():
    raise FileNotFoundError(f"❌ Could not find CSV file at: {csv_path}")
df = pd.read_csv(csv_path)

# === Clean + Predict ===
# Remove rows with missing cluster or features
df = df[df['cluster'] != -1].dropna()

# Ensure consistent features
exclude_cols = ['trial', 'time', 'frame', 'cluster', 'UMAP_1', 'UMAP_2', 'behavior_label', 'source_file']
X_test = df.drop(columns=[col for col in exclude_cols if col in df.columns], errors='ignore')

# Predict behaviors
df['predicted_behavior'] = clf.predict(X_test).astype(str)
print("✅ Behavior predictions added to DataFrame")

# === Map frame to behavior
frame_pred_map = dict(zip(df['frame'], df['predicted_behavior']))

# === Open video ===
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise IOError(f"❌ Could not open video: {video_path}")

fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# === Set up video writer ===
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

# === Optional: color map for better visuals ===
color_map = {
    '0': (0, 255, 0),
    '1': (255, 0, 0),
    '2': (0, 0, 255),
    '3': (255, 255, 0),
    '4': (255, 0, 255),
    '5': (0, 255, 255)
}

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2

# === Process video ===
print("🔄 Writing video with behavior overlay...")
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    label = frame_pred_map.get(frame_idx)
    if label:
        color = color_map.get(label, (255, 255, 255))  # default white
        text = f"Behavior: {label}"
        cv2.putText(frame, text, (50, 60), font, font_scale, color, font_thickness)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

print(f"✅ Done! Labeled video saved to: {output_video_path}")
    