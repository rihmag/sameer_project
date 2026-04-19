import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from ultralytics import YOLO
from weapon_detector import WeaponDetector # Assuming weapon_detector.py is in the same directory
from concurrent.futures import ThreadPoolExecutor

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Weapon & Fight Detector", layout="wide")
st.title("Live Weapon and Fight Detection")

# --- Model Loading (Cached) ---
@st.cache_resource
def load_models():
    with st.spinner("Loading models... This may take a moment."):
        try:
            # Ensure these paths are correct relative to where you run the Streamlit app
            # or provide absolute paths.
            weapon_detector = WeaponDetector(model_path="runs/detect/weapon_yolo_model9/weights/best.pt")
            fight_model = YOLO("runs/detect/fight_detect_model2/weights/best.pt")
            st.success("Models loaded successfully!")
            return weapon_detector, fight_model
        except Exception as e:
            st.error(f"Error loading models. Please check model paths and ensure files exist: {e}")
            st.stop() # Stop the app if models can't be loaded

weapon_detector, fight_model = load_models()

# --- Detection Functions (adapted from both.py) ---
def detect_weapons_wrapper(detector_instance, frame, conf_threshold):
    """Wrapper function for weapon detection with configurable confidence"""
    return detector_instance.detect(frame, conf=conf_threshold, return_classes=True)

def detect_fights_wrapper(model_instance, frame, conf_threshold):
    """Wrapper function for fight detection with configurable confidence"""
    # The YOLO model's call method directly accepts conf
    results = model_instance(frame, conf=conf_threshold, verbose=False) # verbose=False to suppress console output
    fight_classes = []
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = model_instance.names[cls_id]
                fight_classes.append(class_name)
    return results, fight_classes

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")
source_option = st.sidebar.radio("Select Input Source", ("Webcam", "Upload Video"))
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
st.sidebar.markdown("---")

# --- Main Content Area ---
# Initialize session state for webcam control
if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False

# --- Webcam Input Logic ---
if source_option == "Webcam":
    st.header("Webcam Feed")

    col1, col2 = st.columns([1, 1])
    with col1:
        start_button = st.button("Start Webcam Detection", key="start_webcam")
    with col2:
        stop_button = st.button("Stop Webcam Detection", key="stop_webcam")

    if start_button:
        st.session_state.webcam_running = True
    if stop_button:
        st.session_state.webcam_running = False

    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(0) # 0 for default webcam
        if not cap.isOpened():
            st.error("Error: Could not access webcam. Please ensure it's connected and not in use by another application.")
            st.session_state.webcam_running = False # Stop trying
        else:
            st.info("Webcam started. Detecting...")
            st_frame_placeholder = st.empty() # Placeholder for the video feed
            executor = ThreadPoolExecutor(max_workers=2) # Initialize executor for webcam

            while st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to read frame from webcam. Stopping.")
                    st.session_state.webcam_running = False
                    break

                # Parallel detection
                weapon_future = executor.submit(detect_weapons_wrapper, weapon_detector, frame, conf_threshold)
                fight_future = executor.submit(detect_fights_wrapper, fight_model, frame, conf_threshold)

                weapon_results, weapon_classes = weapon_future.result()
                fight_results, fight_classes = fight_future.result()

                # Annotation (similar to both.py)
                # Start with the original frame, then add weapon detections, then fight detections
                annotated_frame = weapon_detector.plot(weapon_results, generic_label=False) # This returns a copy of the frame with weapon annotations

                # Add fight detections on top
                for r in fight_results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls_id = int(box.cls[0])
                            class_name = fight_model.names[cls_id]
                            conf = float(box.conf[0])

                            # Draw red box for fight
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                            # Label with confidence
                            label = f"FIGHT: {class_name} {conf:.2f}"
                            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                            # Label background
                            cv2.rectangle(annotated_frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 0, 255), -1)
                            # Label text
                            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Convert BGR to RGB for Streamlit display
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st_frame_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)

            cap.release()
            executor.shutdown(wait=True)
            st.info("Webcam detection stopped.")
    else:
        st.info("Webcam detection is currently off. Press 'Start Webcam Detection' to begin.")

# --- Upload Video Input Logic ---
elif source_option == "Upload Video":
    st.header("Upload Video File")
    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        # Create a temporary file to save the uploaded video
        # Using a context manager for tempfile ensures it's closed properly
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tfile:
            tfile.write(uploaded_file.read())
            video_path = tfile.name

        st.video(uploaded_file) # Display the original video for context

        st.subheader("Detection Results")
        st_video_frame_placeholder = st.empty() # Placeholder for processed video frames
        progress_bar = st.progress(0)
        status_text = st.empty()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Error: Could not open video file {uploaded_file.name}. Please check the file format.")
            os.unlink(video_path) # Clean up temp file
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0: # Handle cases where frame count might be 0 or unavailable
                st.warning("Could not determine total frames. Progress bar might not be accurate.")
                total_frames = 1 # Avoid division by zero
            frame_count = 0
            executor = ThreadPoolExecutor(max_workers=2) # Initialize executor for video file

            while True:
                ret, frame = cap.read()
                if not ret:
                    status_text.info(f"End of video: {uploaded_file.name}")
                    break

                frame_count += 1
                progress_bar.progress(min(frame_count / total_frames, 1.0))
                status_text.text(f"Processing frame {frame_count}/{total_frames}")

                # Parallel detection
                weapon_future = executor.submit(detect_weapons_wrapper, weapon_detector, frame, conf_threshold)
                fight_future = executor.submit(detect_fights_wrapper, fight_model, frame, conf_threshold)

                weapon_results, weapon_classes = weapon_future.result()
                fight_results, fight_classes = fight_future.result()

                # Annotation (similar to both.py)
                annotated_frame = weapon_detector.plot(weapon_results, generic_label=False)
                for r in fight_results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls_id = int(box.cls[0])
                            class_name = fight_model.names[cls_id]
                            conf = float(box.conf[0])

                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            label = f"FIGHT: {class_name} {conf:.2f}"
                            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(annotated_frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 0, 255), -1)
                            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st_video_frame_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)

            cap.release()
            executor.shutdown(wait=True)
            os.unlink(video_path) # Clean up the temporary file
            st.success("Video processing complete.")
    else:
        st.info("Please upload a video file to start detection.")