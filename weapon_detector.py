from ultralytics import YOLO

class WeaponDetector:
    def __init__(self, model_path="runs/detect/weapon_yolo_model6/weights/best.pt"):
        """Initialize the YOLO weapon detection model."""
        self.model = YOLO(model_path)

    # Inside your WeaponDetector class:

    def detect(self, frame, conf=0.5, return_classes=False):
        """
        Run detection on a frame.
        
        Args:
            frame: Input image/frame
            conf: Confidence threshold
            return_classes: If True, also return list of detected class names
        
        Returns:
            results: YOLO results object
            detected_classes (optional): List of class names detected
        """
        results = self.model.predict(frame, conf=conf)[0]  # Get first result
        
        if return_classes:
            detected_classes = []
            if results.boxes is not None:
                # Get class IDs and convert to names using results.names dict
                class_ids = results.boxes.cls.cpu().tolist()
                for cls_id in class_ids:
                    class_name = results.names[int(cls_id)]
                    detected_classes.append(class_name)
            return results, detected_classes
        
        return results

    def plot(self, results, generic_label=True):
        """
        Plot detection results on frame.
        
        Args:
            results: YOLO results object
            generic_label: If True, use "Weapon" for all labels. 
                        If False, use actual class names from model.
        
        Returns:
            Annotated frame
        """
        if generic_label:
            # Custom annotation with generic "Weapon" label
            annotated_frame = results.orig_img.copy()
            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                
                for box, conf in zip(boxes, confs):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"Weapon: {conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return annotated_frame
        else:
            # Use YOLO's built-in plotting with actual class names
            return results.plot()