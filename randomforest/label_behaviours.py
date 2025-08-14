
import cv2
import pandas as pd
import os
from pathlib import Path

# === Setup ===
video_path = Path(r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\unprocessed\PT_08_Day1\Coins_1\all_3d\2001-01-01\videos-raw\Coins_1-camC.mp4")
output_csv = Path(r"C:\Users\ag05\Desktop\test_data\data\segmentation_data\processed\PT_08_Day1\script_outputs\Coins_1-labelled-output.csv")

# Label map
key_to_label = {
    ord('g'): 'grasp',
    ord('m'): 'move',
    ord('i'): 'idle',
    ord('r'): 'release',
    ord('s'): 'skip',  # skip a frame without labeling
    ord('q'): 'quit'   # quit labeling
}

labeled_data = []

cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print(f"❌ Failed to open video: {video_path}")
    exit()
frame_num = 0

print("\n🎥 Labeling Instructions:")
print(" Press 'g' for Grasp, 'm' for Move, 'i' for Idle, 's' to Skip, 'q' to Quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    label_text = f"Frame {frame_num} - Press key to label"
    cv2.putText(display_frame, label_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    cv2.imshow('Video Labeling', display_frame)
    key = cv2.waitKey(0)  # waits indefinitely for a key press

    if key in key_to_label:
        behavior = key_to_label[key]
        if behavior == 'quit':
            print("❌ Quitting labeling.")
            break
        elif behavior != 'skip':
            labeled_data.append({'frame': frame_num, 'label': behavior})
            print(f"✅ Labeled Frame {frame_num} as '{behavior}'")
    else:
        print("⚠️ Invalid key. Use g/m/i/s/q only.")

    frame_num += 1

cap.release()
cv2.destroyAllWindows()

# Save to CSV
df = pd.DataFrame(labeled_data)
df.to_csv(output_csv, index=False)
print(f"\n✅ Labels saved to {output_csv}")
