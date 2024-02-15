
import numpy as np
from norfair import Detection

def yolo_detections_to_norfair_detections(yolo_detections):
    norfair_detections = []
    boxes = []

    detections_as_xyxy = yolo_detections[0].boxes.xyxy
    for i, det_xyxy in enumerate(detections_as_xyxy):
        det_xyxy = det_xyxy.cpu().numpy()
        bbox = np.array(
            [
                [det_xyxy[0].item(), det_xyxy[1].item()],
                [det_xyxy[2].item(), det_xyxy[3].item()],
            ]
        )
        boxes.append(bbox)

        points = bbox

        norfair_detections.append(
            Detection(points=points)
        )

        
    return norfair_detections, boxes

def _compute_detection_mask(tracked_objects, boxes, frame):
    mask = np.ones(frame.shape[:2], frame.dtype)
    # set to 0 all detections
    for b in boxes:
        i = b.astype(int)
        mask[i[0, 1] : i[1, 1], i[0, 0] : i[1, 0]] = 0
    for obj in tracked_objects:
        i = obj.estimate.astype(int)
        mask[i[0, 1] : i[1, 1], i[0, 0] : i[1, 0]] = 0
        
    return mask

def compute_tracked_objects(results, prev_tracked_objects, frame, motion_estimator, tracker):
    detections, boxes = yolo_detections_to_norfair_detections(results)
    
    # Default argument for tracker.update()
    update_args = {'detections': detections}
    
    if motion_estimator is not None:
        try:
            mask = _compute_detection_mask(prev_tracked_objects, boxes, frame)
            coord_transformations = motion_estimator.update(frame, mask)
            # If motion estimation succeeds, include coord_transformations in update
            update_args['coord_transformations'] = coord_transformations
        except:
            pass
    
    # Update tracker with detections (and coord_transformations if available)
    tracked_objects = tracker.update(**update_args)
    
    return tracked_objects

if __name__ == "__main__":
    pass
