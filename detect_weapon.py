import cv2
from weapon_detector import WeaponDetector
import os
import glob

def main():
    # Initialize the modular detector with your trained weights
    detector = WeaponDetector(model_path="runs/detect/weapon_yolo_model10/weights/best.pt")

    # Define the folder containing video files
    # This assumes 'guns-data' is a sibling directory to detect.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_folder = os.path.join(script_dir, "Guns-dataset")

    # Check if the video folder exists
    if not os.path.exists(video_folder):
        print(f"Error: Video folder '{video_folder}' not found.")
        print("Please ensure the 'guns-data' folder is in the same directory as detect.py.")
        return

    # List all video files in the folder
    # Common video extensions to look for
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.webm']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_folder, ext)))

    if not video_files:
        print(f"No video files found in '{video_folder}' with extensions: {', '.join(video_extensions)}.")
        return

    print(f"Found {len(video_files)} video(s) in '{video_folder}'.")

    for video_path in video_files:
        print(f"\nProcessing video: {os.path.basename(video_path)}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}. Skipping.")
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"End of video: {os.path.basename(video_path)}")
                break # End of current video
            
            # Run weapon detection with a confidence threshold (0.5 = 50% certainty)
            # Now returns: (results, detected_classes) where detected_classes is a list of class names
            results, detected_classes = detector.detect(frame, conf=0.5, return_classes=True)
            
            # Print detected classes for this frame
            if detected_classes:
                print(f"  Frame detections: {', '.join(set(detected_classes))}")

            # Visualize the results on the frame
            # Use generic_label=True for "Weapon" label, or False for specific class names
            annotated_frame = detector.plot(results, generic_label=False)  # Set to False to show actual class names
            
            # Display the frame with a unique window name for each video
            cv2.imshow(f"Weapon Detection - {os.path.basename(video_path)}", annotated_frame)
            
            # Wait for a key press. 'q' to quit all processing, any other key to continue
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # Release the current video capture and close all windows
                cap.release()
                cv2.destroyAllWindows()
                print("Exiting video processing.")
                return # Exit the main function entirely

        cap.release() # Release the video capture object for the current video
        cv2.destroyAllWindows() # Close all OpenCV windows opened for the current video

    print("\nFinished processing all videos.")

if __name__ == "__main__":
    main()