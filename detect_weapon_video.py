import cv2
from weapon_detector import WeaponDetector

def main():
    # Initialize detector
    detector = WeaponDetector(model_path="runs/detect/weapon_yolo_model11/weights/best.pt")

    # Open webcam (0 = default camera)
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

        # Run detection
        results, detected_classes = detector.detect(frame, conf=0.5, return_classes=True)

        # Print detected classes
        if detected_classes:
            print(f"Detected: {', '.join(set(detected_classes))}")

        # Draw detections
        annotated_frame = detector.plot(results, generic_label=False)

        # Show frame
        cv2.imshow("Weapon Detection - Webcam", annotated_frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam stopped.")

if __name__ == "__main__":
    main()