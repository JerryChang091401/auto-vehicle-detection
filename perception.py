# perception.py
from ultralytics import YOLO
import numpy as np

def box_area(box):
    x1, y1, x2, y2 = box[:4]
    return (x2 - x1) * (y2 - y1)

def box_centroid(box):
    x1, y1, x2, y2 = box[:4]
    return ((x1 + x2)/2, (y1 + y2)/2)

class PerceptionSystem:
    """
    Object detection using YOLOv8
    """
    def __init__(self, model_name='yolov8n.pt', device='cpu'):
        self.model = YOLO(model_name)

    def process_frame(self, frame):
        """
        Returns list of detections [x1, y1, x2, y2, conf, cls]
        """
        results = self.model(frame, verbose=False)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf.cpu())
            cls = int(box.cls.cpu())
            detections.append(np.array([x1, y1, x2, y2, conf, cls]))
        return detections
