import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Failed to open camera.")
    exit()

print("Camera started. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not received")
        break

    cv2.imshow("Live Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
