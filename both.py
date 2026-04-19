import cv2
from ultralytics import YOLO
from weapon_detector import WeaponDetector
import os
import glob
from concurrent.futures import ThreadPoolExecutor

def detect_weapons(weapon_detector, frame):
    """Wrapper function for weapon detection"""
    return weapon_detector.detect(frame, conf=0.5, return_classes=True)

def detect_fights(fight_model, frame):
    """Wrapper function for fight detection"""
    results = fight_model(frame, conf=0.5)
    fight_classes = []
    
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = fight_model.names[cls_id]
                fight_classes.append(class_name)
    
    return results, fight_classes

def main():
    # Initialize both detectors
    print("Loading models...")
    
    # Weapon detector (using your modular class)
    weapon_detector = WeaponDetector(
        model_path="runs/detect/weapon_yolo_model11/weights/best.pt"
    )
    
    # Fight detection model (using YOLO directly)
    fight_model = YOLO("runs/detect/fight_detect_model2/weights/best.pt")
    
    print("Models loaded successfully!\n")

    # Folder path for videos
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_folder = os.path.join(script_dir, "self-defence-test")  # ⚠️ Change this to your folder name

    # Check folder
    if not os.path.exists(video_folder):
        print(f"Error: Video folder '{video_folder}' not found.")
        print("Please ensure the video folder exists in the same directory as the script.")
        return

    # Collect all video files
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.webm']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_folder, ext)))

    if not video_files:
        print(f"No video files found in '{video_folder}'.")
        return

    print(f"Found {len(video_files)} video(s) in '{video_folder}'.\n")

    # Create thread pool (reused across all videos for efficiency)
    # max_workers=2 since we have exactly 2 detection tasks
    executor = ThreadPoolExecutor(max_workers=2)

    # Process each video
    for video_path in video_files:
        print(f"Processing: {os.path.basename(video_path)}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open {video_path}. Skipping.")
            continue

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"  End of video: {os.path.basename(video_path)}")
                break

            frame_count += 1

            # ========== PARALLEL DETECTION ==========
            # Submit both detection tasks concurrently
            weapon_future = executor.submit(detect_weapons, weapon_detector, frame)
            fight_future = executor.submit(detect_fights, fight_model, frame)
            
            # Wait for both to complete and get results
            weapon_results, weapon_classes = weapon_future.result()
            fight_results, fight_classes = fight_future.result()

            # ========== COMBINE RESULTS ==========
            all_detections = []
            
            # Annotate weapon detections (original detector colors)
            weapon_annotated = weapon_detector.plot(
                weapon_results, 
                generic_label=False
            )
            
            # Annotate fight detections on top (red color for fights)
            fight_annotated = weapon_annotated.copy()
            
            for r in fight_results:
                if r.boxes is not None:
                    for box in r.boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        class_name = fight_model.names[cls_id]
                        conf = float(box.conf[0])
                        
                        # Draw red box for fight
                        cv2.rectangle(fight_annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        
                        # Label with confidence
                        label = f"FIGHT: {class_name} {conf:.2f}"
                        (text_w, text_h), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                        )
                        
                        # Label background
                        cv2.rectangle(
                            fight_annotated, 
                            (x1, y1 - text_h - 10), 
                            (x1 + text_w, y1), 
                            (0, 0, 255), 
                            -1
                        )
                        # Label text
                        cv2.putText(
                            fight_annotated, 
                            label, 
                            (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, 
                            (255, 255, 255), 
                            2
                        )

            # Combine detection lists for console output
            if weapon_classes:
                all_detections.extend([f"[WEAPON] {c}" for c in set(weapon_classes)])
            if fight_classes:
                all_detections.extend([f"[FIGHT] {c}" for c in set(fight_classes)])

            # Print detections every 30 frames (reduce console spam)
            if all_detections and frame_count % 30 == 0:
                print(f"  Frame {frame_count}: {', '.join(all_detections)}")

            # Display combined results
            window_name = f"Weapon & Fight Detection - {os.path.basename(video_path)}"
            cv2.imshow(window_name, fight_annotated)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                executor.shutdown(wait=False)  # Clean shutdown
                print("\nExiting...")
                return

        cap.release()
        cv2.destroyAllWindows()

    # Clean shutdown of thread pool
    executor.shutdown(wait=True)
    print("\n✅ Finished processing all videos.")

if __name__ == "__main__":
    main()