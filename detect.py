import cv2
from weapon_detector import WeaponDetector

def main():
    # Initialize the modular detector with your trained weights
    detector = WeaponDetector(model_path="runs/detect/weapon_yolo_model6/weights/best.pt")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run weapon detection with a confidence threshold (0.5 = 50% certainty)
        results = detector.detect(frame, conf=0.5)

        # Visualize the results on the frame
        annotated_frame = detector.plot(results)
        
        cv2.imshow("Weapon Detection", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()