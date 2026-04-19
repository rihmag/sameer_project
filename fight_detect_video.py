import cv2
from ultralytics import YOLO
import os
import glob

def main():
    # Load fight detection model
    model = YOLO("runs/detect/fight_detect_model2/weights/best.pt")

    # Folder path (same structure as your weapon script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_folder = os.path.join(script_dir, "self_defence_test")  # ⚠️ change if needed

    # Check folder
    if not os.path.exists(video_folder):
        print(f"Error: Video folder '{video_folder}' not found.")
        return

    # Collect videos
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.webm']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_folder, ext)))

    if not video_files:
        print("No videos found.")
        return

    print(f"Found {len(video_files)} video(s).")

    # Loop through each video
    for video_path in video_files:
        print(f"\nProcessing: {os.path.basename(video_path)}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error opening {video_path}")
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"End of video: {os.path.basename(video_path)}")
                break

            # Run YOLO
            results = model(frame, conf=0.5)

            detected_classes = []

            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        class_name = model.names[cls_id]
                        detected_classes.append(class_name)

            # Print detections
            if detected_classes:
                print(f"  Frame detections: {', '.join(set(detected_classes))}")

            # Draw boxes
            annotated_frame = results[0].plot()

            # Show frame (same style as yours)
            cv2.imshow(f"Fight Detection - {os.path.basename(video_path)}", annotated_frame)

            # Exit logic (same as yours)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print("Exiting...")
                return

        cap.release()
        cv2.destroyAllWindows()

    print("\nFinished processing all videos.")

if __name__ == "__main__":
    main()