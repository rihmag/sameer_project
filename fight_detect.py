# script for fight detect 
import cv2
from ultralytics import YOLO

def main():
    # Load your trained fight detection model
    model = YOLO("runs/detect/fight_detect_model2/weights/best.pt")  
    # ⚠️ change path if needed

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    print("Webcam started. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        # Run inference
        results = model(frame, conf=0.5)

        detected_classes = []

        # Extract detected class names
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    detected_classes.append(class_name)

        # Print detections
        if detected_classes:
            print(f"Detected: {', '.join(set(detected_classes))}")

        # Annotate frame
        annotated_frame = results[0].plot()

        # Show output
        cv2.imshow("Fight Detection - Webcam", annotated_frame)

        # Exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam stopped.")

if __name__ == "__main__":
    main()
