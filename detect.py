import cv2
from ultralytics import YOLO

# Loading a fresh, pre-trained YOLOv8 nano model
weapon_model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run weapon detection
    results = weapon_model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    
    cv2.imshow("Weapon Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()