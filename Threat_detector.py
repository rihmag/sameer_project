import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
from collections import deque
from enum import Enum

# ─── Threat levels ────────────────────────────────────────────────────────────
class ThreatLevel(Enum):
    NONE     = ("NONE",     (80, 200, 80),   0)
    CAUTION  = ("CAUTION",  (0, 200, 255),   1)
    HIGH     = ("HIGH",     (0, 140, 255),   2)
    CRITICAL = ("CRITICAL", (0, 0, 220),     3)

# ─── Landmark indices (MediaPipe Pose) ────────────────────────────────────────
WRISTS      = [15, 16]
ELBOWS      = [13, 14]
SHOULDERS   = [11, 12]
HIPS        = [23, 24]
KNEES       = [25, 26]
ANKLES      = [27, 28]
NOSE        = 0

@dataclass
class FrameState:
    threat:       ThreatLevel = ThreatLevel.NONE
    reasons:      list = None
    weapon_boxes: list = None

    def __post_init__(self):
        self.reasons      = self.reasons      or []
        self.weapon_boxes = self.weapon_boxes or []


class ActionAnalyser:
    """
    Detects threatening body actions using MediaPipe landmark geometry.
    All checks are purely geometric — no ML classifier needed.
    """

    def __init__(self, history_len: int = 12):
        # Rolling buffer of landmark positions (for velocity / motion checks)
        self.history = deque(maxlen=history_len)

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _lm(landmarks, idx):
        """Return (x, y, visibility) for a single landmark index."""
        lm = landmarks.landmark[idx]
        return np.array([lm.x, lm.y]), lm.visibility

    @staticmethod
    def _dist(a, b):
        return float(np.linalg.norm(a - b))

    @staticmethod
    def _angle(a, b, c):
        """Angle at vertex b formed by a-b-c (degrees)."""
        ba = a - b
        bc = c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

    def _shoulder_width(self, landmarks):
        ls, _ = self._lm(landmarks, SHOULDERS[0])
        rs, _ = self._lm(landmarks, SHOULDERS[1])
        return self._dist(ls, rs) + 1e-6       # normalisation baseline

    # ── individual threat checks ──────────────────────────────────────────────
    def _check_raised_striking_arm(self, landmarks, sw):
        """
        Wrist raised well above shoulder + elbow bent:
        classic overhead strike or punch preparation.
        """
        reasons = []
        for wrist_i, elbow_i, shoulder_i in zip(WRISTS, ELBOWS, SHOULDERS):
            w, wv = self._lm(landmarks, wrist_i)
            e, ev = self._lm(landmarks, elbow_i)
            s, sv = self._lm(landmarks, shoulder_i)
            if min(wv, ev, sv) < 0.5:
                continue
            # In MediaPipe normalised coords y=0 is top → smaller y = higher
            wrist_above_shoulder = (s[1] - w[1]) / sw   # positive = raised
            elbow_angle          = self._angle(w, e, s)
            # Raised + arm not fully extended (bent = active motion)
            if wrist_above_shoulder > 0.15 and elbow_angle < 150:
                reasons.append("raised/striking arm posture")
        return reasons

    def _check_lunging_stance(self, landmarks, sw):
        """
        One knee deeply bent while the other leg is extended:
        lunging / charging motion.
        """
        reasons = []
        lk, lkv = self._lm(landmarks, KNEES[0])
        rk, rkv = self._lm(landmarks, KNEES[1])
        lh, lhv = self._lm(landmarks, HIPS[0])
        rh, rhv = self._lm(landmarks, HIPS[1])
        la, lav = self._lm(landmarks, ANKLES[0])
        ra, rav = self._lm(landmarks, ANKLES[1])
        if min(lkv, rkv, lhv, rhv, lav, rav) < 0.45:
            return reasons
        # Hip-to-ankle horizontal spread (lunge = wide stance)
        hip_mid    = (lh + rh) / 2
        ankle_spread = abs(la[0] - ra[0]) / sw
        # Asymmetric knee height = one leg bent deeply
        knee_height_diff = abs(lk[1] - rk[1]) / sw
        if ankle_spread > 1.2 and knee_height_diff > 0.2:
            reasons.append("lunging / charging stance")
        return reasons

    def _check_grabbing_reach(self, landmarks, sw):
        """
        Both arms extended forward at roughly the same height:
        grabbing, choking, or shoving posture.
        """
        reasons = []
        lw, lwv = self._lm(landmarks, WRISTS[0])
        rw, rwv = self._lm(landmarks, WRISTS[1])
        ls, lsv = self._lm(landmarks, SHOULDERS[0])
        rs, rsv = self._lm(landmarks, SHOULDERS[1])
        if min(lwv, rwv, lsv, rsv) < 0.5:
            return reasons
        # Both wrists ahead of (lower y than) their respective shoulders
        l_reach = (ls[1] - lw[1]) / sw   # positive = wrist above shoulder
        r_reach = (rs[1] - rw[1]) / sw
        wrist_height_diff = abs(lw[1] - rw[1]) / sw
        if l_reach > -0.05 and r_reach > -0.05 and wrist_height_diff < 0.25:
            reasons.append("bilateral arm reach (grabbing / choking posture)")
        return reasons

    def _check_arm_velocity(self, landmarks):
        """
        Sudden large wrist displacement between frames:
        fast swing / stab / slash motion.
        """
        reasons = []
        if len(self.history) < 4:
            return reasons
        current_wrists = [self._lm(landmarks, i)[0] for i in WRISTS]
        prev_landmarks = self.history[-4]
        prev_wrists    = [self._lm(prev_landmarks, i)[0] for i in WRISTS]
        for cw, pw in zip(current_wrists, prev_wrists):
            if self._dist(cw, pw) > 0.18:    # normalised-coord threshold
                reasons.append("rapid arm motion (swing / stab)")
                break
        return reasons

    def _check_body_proximity(self, landmarks_list):
        """
        Two or more people: check if any person's wrist is very close
        to another person's head / torso — physical contact threat.
        """
        reasons = []
        if len(landmarks_list) < 2:
            return reasons
        positions = []
        for lm in landmarks_list:
            wrists = [self._lm(lm, i)[0] for i in WRISTS]
            nose,  nv = self._lm(lm, NOSE)
            lhip,  _  = self._lm(lm, HIPS[0])
            rhip,  _  = self._lm(lm, HIPS[1])
            torso_mid = (lhip + rhip) / 2
            positions.append({"wrists": wrists, "head": nose, "torso": torso_mid})
        for i, p1 in enumerate(positions):
            for j, p2 in enumerate(positions):
                if i == j:
                    continue
                for w in p1["wrists"]:
                    if self._dist(w, p2["head"])  < 0.12 or \
                       self._dist(w, p2["torso"]) < 0.10:
                        reasons.append("hand-to-body contact threat between persons")
                        return reasons   # one instance is enough
        return reasons

    # ── public API ────────────────────────────────────────────────────────────
    def analyse(self, landmarks_list):
        """
        Run all checks on a list of detected person landmarks.
        Returns a list of threat reason strings.
        """
        all_reasons = []
        for lm in landmarks_list:
            sw = self._shoulder_width(lm)
            all_reasons += self._check_raised_striking_arm(lm, sw)
            all_reasons += self._check_lunging_stance(lm, sw)
            all_reasons += self._check_grabbing_reach(lm, sw)
            all_reasons += self._check_arm_velocity(lm)
            # Store for next-frame velocity check
            self.history.append(lm)
        all_reasons += self._check_body_proximity(landmarks_list)
        return list(set(all_reasons))   # deduplicate


class WeaponProximityChecker:
    """
    Checks whether a detected weapon bounding box overlaps or is
    near a person's hand/wrist landmarks.
    """

    @staticmethod
    def check(weapon_boxes, landmarks_list, frame_w, frame_h):
        """
        weapon_boxes : list of (x1,y1,x2,y2) in pixel coords
        landmarks_list: list of mediapipe landmark objects
        Returns list of reason strings.
        """
        reasons = []
        if not weapon_boxes or not landmarks_list:
            return reasons
        for lm in landmarks_list:
            for wrist_i in WRISTS:
                wrist_lm = lm.landmark[wrist_i]
                if wrist_lm.visibility < 0.5:
                    continue
                wx = int(wrist_lm.x * frame_w)
                wy = int(wrist_lm.y * frame_h)
                for (x1, y1, x2, y2) in weapon_boxes:
                    # Expand box slightly for "near" check
                    pad = int((x2 - x1) * 0.4)
                    if (x1 - pad) < wx < (x2 + pad) and \
                       (y1 - pad) < wy < (y2 + pad):
                        reasons.append("weapon in hand / gripped")
                        break
        return list(set(reasons))


class ThreatScorer:
    """
    Converts a list of threat reasons into a ThreatLevel enum value.
    Uses a simple additive scoring model — tune weights as needed.
    """

    WEIGHTS = {
        "weapon detected":                          3,
        "weapon in hand / gripped":                 4,
        "raised/striking arm posture":              2,
        "rapid arm motion (swing / stab)":          3,
        "lunging / charging stance":                2,
        "bilateral arm reach (grabbing / choking posture)": 2,
        "hand-to-body contact threat between persons":      4,
    }

    def score(self, reasons):
        total = sum(self.WEIGHTS.get(r, 1) for r in reasons)
        if total == 0:
            return ThreatLevel.NONE
        elif total <= 2:
            return ThreatLevel.CAUTION
        elif total <= 5:
            return ThreatLevel.HIGH
        else:
            return ThreatLevel.CRITICAL


class ThreatDetector:
    """
    Main pipeline: combines YOLO weapon detection + MediaPipe pose analysis.
    """

    def __init__(self, weapon_model_path: str, conf: float = 0.5):
        print("[INFO] Loading weapon model …")
        self.weapon_model = YOLO(weapon_model_path)
        self.conf = conf

        print("[INFO] Initialising MediaPipe …")
        mp_holistic = mp.solutions.holistic
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.action_analyser  = ActionAnalyser()
        self.proximity_checker = WeaponProximityChecker()
        self.scorer           = ThreatScorer()
        self.mp_draw          = mp.solutions.drawing_utils
        self.mp_styles        = mp.solutions.drawing_styles

    # ── detection ─────────────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> FrameState:
        h, w = frame.shape[:2]
        state = FrameState()

        # 1 ── Weapon detection (YOLO) ─────────────────────────────────────────
        yolo_results = self.weapon_model(frame, conf=self.conf, verbose=False)
        for result in yolo_results:
            if result.boxes is not None:
                for cls, box in zip(result.boxes.cls, result.boxes.xyxy):
                    class_id = int(cls)

                    if class_id in [0, 1]:   # 0 = gun, 1 = knife
                        state.reasons.append("weapon detected")
                        state.weapon_boxes.append(tuple(map(int, box.cpu().numpy())))

        # 2 ── Pose detection (MediaPipe) ──────────────────────────────────────
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_result = self.holistic.process(rgb)

        # Collect all detected persons' landmarks
        # (Holistic detects one person; for multi-person extend here)
        landmarks_list = []
        if mp_result.pose_landmarks:
            landmarks_list.append(mp_result.pose_landmarks)

        # 3 ── Action analysis ─────────────────────────────────────────────────
        state.reasons += self.action_analyser.analyse(landmarks_list)

        # 4 ── Weapon proximity ────────────────────────────────────────────────
        state.reasons += self.proximity_checker.check(
            state.weapon_boxes, landmarks_list, w, h
        )

        # 5 ── Score ───────────────────────────────────────────────────────────
        state.threat = self.scorer.score(state.reasons)
        return state, mp_result

    # ── annotation ────────────────────────────────────────────────────────────
    def annotate(self, frame: np.ndarray, state: FrameState, mp_result) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        # Draw pose skeleton
        if mp_result.pose_landmarks:
            self.mp_draw.draw_landmarks(
                out,
                mp_result.pose_landmarks,
                mp.solutions.holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_draw.DrawingSpec(
                    color=(220, 220, 220), thickness=1, circle_radius=2
                ),
                connection_drawing_spec=self.mp_draw.DrawingSpec(
                    color=(180, 180, 180), thickness=1
                ),
            )

        # Draw weapon bounding boxes
        for (x1, y1, x2, y2) in state.weapon_boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 80, 220), 2)
            cv2.putText(out, "WEAPON", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 220), 2)

        # Threat level banner (top of frame)
        level_name, bgr, _ = state.threat.value
        banner_h = 50
        overlay  = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), bgr, -1)
        cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
        cv2.putText(out, f"THREAT: {level_name}", (12, 34),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

        # Reason list (bottom-left)
        for i, reason in enumerate(state.reasons[:6]):   # cap at 6 lines
            y = h - 20 - i * 22
            cv2.putText(out, f"• {reason}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

        return out

    def close(self):
        self.holistic.close()


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    MODEL_PATH = "runs/detect/weapon_yolo_model7/weights/best.pt"
    SOURCE     = 0    # 0 = webcam | or "path/to/video.mp4"

    detector = ThreatDetector(weapon_model_path=MODEL_PATH, conf=0.5)
    cap      = cv2.VideoCapture(SOURCE)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {SOURCE}")
        return

    print("[INFO] Running — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        state, mp_result = detector.process_frame(frame)
        annotated        = detector.annotate(frame, state, mp_result)
        cv2.imshow("Threat Detection", annotated)

        # Console alert for high-severity threats
        if state.threat in (ThreatLevel.HIGH, ThreatLevel.CRITICAL):
            print(f"[ALERT] {state.threat.value[0]} — {', '.join(state.reasons)}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()