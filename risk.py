# risk.py
from perception import box_area, box_centroid

class RiskEstimator:
    """
    Motion-based collision risk estimator.
    Uses bounding-box growth and downward motion to assign risk.
    """
    def __init__(self, growth_thresholds=(0.25, 0.1)):
        self.prev_areas = {}       # stores previous area per object ID
        self.prev_positions = {}   # stores previous centroid per object ID
        self.high_thresh, self.mid_thresh = growth_thresholds

    def compute_risks(self, detections):
        """
        Returns list of risk levels ('GREEN','YELLOW','RED')
        for each detection.
        """
        risks = []

        for i, det in enumerate(detections):
            area = box_area(det)
            cx, cy = box_centroid(det)
            obj_id = i  # use index as simple object ID

            if obj_id in self.prev_areas:
                prev_area = self.prev_areas[obj_id]
                prev_cy = self.prev_positions[obj_id][1]

                growth = (area - prev_area) / max(prev_area, 1)
                moving_closer = cy > prev_cy

                if growth > self.high_thresh and moving_closer:
                    risk = "RED"
                elif growth > self.mid_thresh:
                    risk = "YELLOW"
                else:
                    risk = "GREEN"
            else:
                risk = "GREEN"

            # Update history
            self.prev_areas[obj_id] = area
            self.prev_positions[obj_id] = (cx, cy)
            risks.append(risk)

        return risks
