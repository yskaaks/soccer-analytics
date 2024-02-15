
def compute_detected_objects(yolo_detections, frame):
    detected_objects = []
    
    detections_as_xyxy = yolo_detections[0].boxes.xyxy
    detections_ids = [0 for _ in detections_as_xyxy]
    
    for det_xyxy, det_id in zip(detections_as_xyxy, detections_ids):
        det_xyxy = det_xyxy.cpu().numpy()
        
        det_object = DetectedObject(det_xyxy, det_id)
        
        detected_objects.append(det_object)
    
    return detected_objects


class DetectedObject:
    def __init__(self, bbox, obj_id):
        self.bbox = bbox
        self.id = obj_id
        