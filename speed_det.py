# speed_up_det.py
import time
from perception import PerceptionSystem
from risk import RiskEstimator

class SpeedUpDetector:
    def __init__(self, skip_frames=10, resize_width=None):
        self.perception = PerceptionSystem()
        self.risk_estimator = RiskEstimator()
        self.skip_frames = skip_frames
        self.resize_width = resize_width
        self.detections = []
        self.risks = []
        self.frame_count = 0
        self.prev_time = time.time()

    def process_frame(self, frame):
        self.frame_count += 1
        orig_h, orig_w = frame.shape[:2]
        frame_proc = frame

        # Resize for faster detection if requested
        scale_ratio = 1.0
        if self.resize_width:
            scale_ratio = self.resize_width / orig_w
            new_h = int(orig_h * scale_ratio)
            import cv2
            frame_proc = cv2.resize(frame, (self.resize_width, new_h))

        # Run YOLO only every skip_frames
        if self.frame_count % self.skip_frames == 0:
            self.detections = self.perception.process_frame(frame_proc)
            # Scale boxes back to original frame size
            for det in self.detections:
                det[[0,2]] = det[[0,2]] / scale_ratio
                det[[1,3]] = det[[1,3]] / scale_ratio
            self.risks = self.risk_estimator.compute_risks(self.detections)

        # Draw overlays on frame
        frame_out = self.draw_detections(frame.copy(), self.detections, self.risks)

        # FPS calculation
        curr_time = time.time()
        fps = 1.0 / max(curr_time - self.prev_time, 1e-6)
        self.prev_time = curr_time

        return frame_out, self.detections, self.risks, fps

    @staticmethod
    def draw_detections(frame, detections, risks):
        import cv2
        color_map = {"GREEN": (0, 255, 0), "YELLOW": (0, 255, 255), "RED": (0, 0, 255)}
        for det, risk in zip(detections, risks):
            x1, y1, x2, y2, conf, cls = det
            color = color_map[risk]
            label = f"{risk} ({conf:.2f})"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1)-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame
