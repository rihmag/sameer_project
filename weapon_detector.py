from ultralytics import YOLO

class WeaponDetector:
    def __init__(self, model_path="runs/detect/weapon_yolo_model6/weights/best.pt"):
        """Initialize the YOLO weapon detection model."""
        self.model = YOLO(model_path)

    def detect(self, frame, conf=0.5):
        """Perform detection on the provided frame."""
        return self.model(frame, conf=conf)

    def plot(self, results):
        """Return an annotated frame from the detection results."""
        return results[0].plot()