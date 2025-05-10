import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

model = YOLO("best.pt")
tracker = Sort()
trajectories = {}

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open camera.")
    exit()

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
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for i in range(1, len(trajectories[track_id])):
            pt1 = trajectories[track_id][i - 1]
            pt2 = trajectories[track_id][i]
            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

    cv2.imshow("Drone Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
