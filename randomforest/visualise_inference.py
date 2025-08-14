import cv2
import pandas as pd

# Paths
inference_file = r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\processed\PT_15_Day1\script_outputs\third_iteration_prediction.csv"
video_file = r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\unprocessed\PT_15_Day1\Tapes_5\all_3d\2001-01-01\videos-raw\Tapes_5-camB.mp4"

# Load predictions and filter for trial 5
df_pred = pd.read_csv(inference_file)
trial_number = 5
df_trial = df_pred[df_pred['trial'] == trial_number]

# Create a dict/frame lookup for predicted_label or fallback to label
label_lookup = df_trial.set_index('frame')['predicted_label'].fillna(df_trial.set_index('frame')['label']).to_dict()

# Open video
cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
color = (0, 255, 0)  # Green
thickness = 2
position = (10, 30)  # Top-left corner

frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Compute the frame bin start (e.g., 0, 30, 60, ...)
    frame_bin = (frame_num // 30) * 30

    # Get label for the current bin
    label = label_lookup.get(frame_bin, "No label")

    # Overlay text on frame
    cv2.putText(frame, f"Trial {trial_number}: {label}", position, font, font_scale, color, thickness)

    # Show frame
    cv2.imshow('Labeled Video', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    frame_num += 1

cap.release()
cv2.destroyAllWindows()
