import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import os

VIDEO_PATH = "input/drone_video.mp4"
OUTPUT_PATH = "output/result.mp4"
MODEL_PATH = "best.pt"

model = YOLO(MODEL_PATH)
tracker = Sort()
trajectories = {}

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Failed to open video file.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

os.makedirs("output", exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        if conf > 0.3:
            detections.append([x1, y1, x2, y2, conf, 0])

    detections = np.array(detections)
    tracked = tracker.update(detections)

    for obj in tracked:
        x1, y1, x2, y2, track_id = map(int, obj[:5])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        if track_id not in trajectories:
            trajectories[track_id] = []
        trajectories[track_id].append((cx, cy))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {int(track_id)}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for point_idx in range(1, len(trajectories[track_id])):
            pt1 = trajectories[track_id][point_idx - 1]
            pt2 = trajectories[track_id][point_idx]
            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

    out.write(frame)

cap.release()
out.release()
print("Done! Saved to output/result.mp4")
