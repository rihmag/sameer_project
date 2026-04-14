from ultralytics import YOLO

class WeaponDetector:
    def __init__(self, model_path="runs/detect/weapon_yolo_model6/weights/best.pt"):
        """Initialize the YOLO weapon detection model."""
        self.model = YOLO(model_path)

    def detect(self, frame, conf=0.5):
        """Perform detection on the provided frame."""
        return self.model(frame, conf=conf)

    def plot(self, results, generic_label=False):
        """Return an annotated frame from the detection results."""
        if generic_label:
            # Override the names dictionary in the results object
            # mapping every class ID to the generic "Weapon" label.
            for r in results:
                r.names = {i: "Weapon" for i in r.names}
        return results[0].plot()