

from ultralytics import YOLO
import cv2
from tqdm import tqdm
import numpy as np
import os
import json

# Color segmentation code -----------------------------------------------------
def save_hsv_ranges(hsv_ranges, file_path):
    # Convert numpy arrays to lists for JSON serialization
    hsv_ranges_list = [(lower.tolist(), upper.tolist()) for lower, upper in hsv_ranges]
    with open(file_path, 'w') as file:
        json.dump(hsv_ranges_list, file)

def load_hsv_ranges(file_path):
    with open(file_path, 'r') as file:
        hsv_ranges_list = json.load(file)
    # Convert lists back to numpy arrays
    hsv_ranges = [(np.array(lower), np.array(upper)) for lower, upper in hsv_ranges_list]
    return hsv_ranges

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_index = total_frames // 2

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return None, None

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
    ret, middle_frame = cap.read()
    if not ret:
        print("Error: Could not read the middle frame.")
        return None, None

    cap.release()
    return first_frame, middle_frame

def scale_and_concat(frames):
    scaled_frames = [cv2.resize(frame, (frame.shape[1] // 3, frame.shape[0] // 3)) for frame in frames]
    concatenated_frame = np.concatenate(scaled_frames, axis=1)
    return concatenated_frame

def setup_hsv_ranges(config, n_classes=2):
    if config['hsv_ranges_path'] and os.path.exists(config['hsv_ranges_path']):
        # Load HSV ranges from the specified file
        return load_hsv_ranges(config['hsv_ranges_path'])
    
    else:
        # Proceed with manual setup of HSV ranges
        first_frame, middle_frame = extract_frames(config['input_video_path'])
        if first_frame is None or middle_frame is None:
            return
    
        concatenated_frame = scale_and_concat([first_frame, middle_frame])
        segmentation = VideoSegmentation('Segmentation')
        hsv_ranges_or_nans = []
        
        for _ in range(n_classes):
            while True:
                segmentation.update_segmentation(concatenated_frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('y'):
                    hsv_ranges_or_nans.append(segmentation.get_hsv_ranges())
                    break
                elif key == ord('n'):
                    hsv_ranges_or_nans.append(None)
                    break
                
            segmentation.reset_trackbars()
    
        cv2.destroyAllWindows()
        
        # Save the manually setup HSV ranges to a file in the 'outputs' directory
        save_hsv_ranges(hsv_ranges_or_nans, config['hsv_ranges_path'])
        
        return hsv_ranges_or_nans

class VideoSegmentation:
    def __init__(self, window_name):
        self.window_name = window_name
        cv2.namedWindow(self.window_name)
        self.lower_bound = np.array([0, 0, 0])
        self.upper_bound = np.array([179, 255, 255])
        self.create_trackbars()

    def create_trackbars(self):
        """Creates trackbars for HSV range selection."""
        cv2.createTrackbar('H Min', self.window_name, 0, 179, self.noop)
        cv2.createTrackbar('H Max', self.window_name, 179, 179, self.noop)
        cv2.createTrackbar('S Min', self.window_name, 0, 255, self.noop)
        cv2.createTrackbar('S Max', self.window_name, 255, 255, self.noop)
        cv2.createTrackbar('V Min', self.window_name, 0, 255, self.noop)
        cv2.createTrackbar('V Max', self.window_name, 255, 255, self.noop)

    def noop(self, x):
        """No-operation function for trackbar callback."""
        pass

    def get_hsv_ranges(self):
        """Returns the current HSV range selections."""
        
        return self.lower_bound, self.upper_bound
    
    def update_segmentation(self, concatenated_frame):
        """Updates the segmentation based on trackbar positions."""
        hsv = cv2.cvtColor(concatenated_frame, cv2.COLOR_BGR2HSV)
        self.lower_bound = np.array([cv2.getTrackbarPos('H Min', self.window_name), 
                                     cv2.getTrackbarPos('S Min', self.window_name), 
                                     cv2.getTrackbarPos('V Min', self.window_name)])
        self.upper_bound = np.array([cv2.getTrackbarPos('H Max', self.window_name), 
                                     cv2.getTrackbarPos('S Max', self.window_name), 
                                     cv2.getTrackbarPos('V Max', self.window_name)])
        mask = cv2.inRange(hsv, self.lower_bound, self.upper_bound)        
        segmented = cv2.bitwise_and(concatenated_frame, concatenated_frame, mask=mask)
        cv2.imshow(self.window_name, segmented)
        return self.lower_bound, self.upper_bound
    
    def reset_trackbars(self):
        """Resets all trackbars to their initial values."""
        cv2.setTrackbarPos('H Min', self.window_name, 0)
        cv2.setTrackbarPos('H Max', self.window_name, 179)
        cv2.setTrackbarPos('S Min', self.window_name, 0)
        cv2.setTrackbarPos('S Max', self.window_name, 255)
        cv2.setTrackbarPos('V Min', self.window_name, 0)
        cv2.setTrackbarPos('V Max', self.window_name, 255)

# Object detection functions --------------------------------------------------
class DetectedObject:
    def __init__(self, bbox, obj_id):
        self.bbox = bbox
        self.id = obj_id
        
class DetectionProcessor:
    def __init__(self, model_path, classes_hsv_ranges):
        self.model = YOLO(model_path)
        self.classes_hsv_ranges = classes_hsv_ranges
        self.n_classes = len(self.classes_hsv_ranges)

    def compute_detected_objects(self, yolo_detections, frame):
        detected_objects = []
        detections_as_xyxy = yolo_detections[0].boxes.xyxy
        # detections_ids = [0 for _ in detections_as_xyxy]
        
        # for det_xyxy, det_id in zip(detections_as_xyxy, detections_ids):
        for det_xyxy in detections_as_xyxy:
            det_xyxy = det_xyxy.cpu().numpy()
            
            x1, y1, x2, y2 = det_xyxy
            frame_crop = frame[int(y1):int(y2)+1, int(x1):int(x2)+1, :]
            det_id = self.predict_class_by_color(frame_crop)
            
            det_object = DetectedObject(det_xyxy, det_id)
            detected_objects.append(det_object)
        
        return detected_objects
    
    def predict_class_by_color(self, frame_crop):
        frame_crop_hsv = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2HSV)
        class_scores = np.zeros((self.n_classes,), dtype=np.float32)

        for i, hsv_ranges in enumerate(self.classes_hsv_ranges):
            lower_bound, upper_bound = hsv_ranges
            mask = cv2.inRange(frame_crop_hsv, lower_bound, upper_bound)

            class_scores[i] = np.sum(mask)

        return np.argmax(class_scores) + 1


    def id_to_color(self, box_id):
        # Example strategy: cycle through a list of predefined BGR colors
        colors = [
            (0, 255, 0),  # Green
            (0, 0, 255),  # Red
            (255, 0, 0),  # Blue
            (0, 255, 255), # Cyan
            (255, 0, 255), # Magenta
            (255, 255, 0)  # Yellow
        ]
        # Use box_id to select a color, ensuring it cycles through the list
        return colors[box_id % len(colors)]    

    def draw_detected_objects(self, original_frame, detected_objects):
        frame = original_frame.copy()
        
        # Fixed dimensions for the triangle
        tri_height = 25
        tri_base_half_length = 15
        vertical_offset = 20
        
        for detected_obj in detected_objects:
            x1, y1, x2, y2 = detected_obj.bbox
            
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            
            width, height = (x2 - x1), (y2 - y1)
            center = (x1 + width/2, y1 + height/2)
            
            box_id = detected_obj.id
            
            circle_color = self.id_to_color(box_id)
            
            # Calculate the bottom center of the bounding box for the ellipse
            bottom_center = (int(center[0]), int(center[1] + height / 2))
            
            # Define the axes for the ellipse and angle
            ellipse_axes = (int(width / 2), int(height / 10))
            ellipse_angle = 0
            ellipse_thickness = 4

            # Draw the bottom half ellipse in blue with the specified thickness
            cv2.ellipse(frame, bottom_center, ellipse_axes, ellipse_angle, 0, 180, circle_color, ellipse_thickness)

            # Calculate the bottom point of the triangle (above the bounding box)
            top_point_triangle = (int(center[0]), int(center[1] - height / 2) - vertical_offset)

            # Triangle points
            p1 = (top_point_triangle[0], top_point_triangle[1] + tri_height)
            p2 = (top_point_triangle[0] - tri_base_half_length, top_point_triangle[1])
            p3 = (top_point_triangle[0] + tri_base_half_length, top_point_triangle[1])

            # Draw the filled triangle in white for the ID
            cv2.drawContours(frame, [np.array([p1, p2, p3])], 0, (255, 255, 255), -1)

            # Add the ID text in black, centered in the triangle
            text_size = cv2.getTextSize(str(box_id), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = p1[0] - text_size[0] // 2
            text_y = p1[1] - 2* text_size[1] // 3
            cv2.putText(frame, str(box_id), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return frame

# General video processing functions ------------------------------------------

class VideoProcessor:
    def __init__(self, config, classes_hsv_ranges):
        self.config = config
        self.detection_processor = DetectionProcessor(config["yolo_model_path"], classes_hsv_ranges)
        
    def process_frame(self, frame):
        detections = self.detection_processor.model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
        detected_objects = self.detection_processor.compute_detected_objects(detections, frame)
        frame_with_detections = self.detection_processor.draw_detected_objects(frame, detected_objects)
        
        return frame_with_detections
        
    def process_video(self):
        cap = cv2.VideoCapture(self.config['input_video_path'])
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with tqdm(total=total_frames, desc="Processing video frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    processed_frame = self.process_frame(frame)
                    
                    cv2.imshow("Detections", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    pbar.update(1)
                else:
                    break
                    
        cv2.destroyAllWindows()
        
# Utility functions -----------------------------------------------------------
def create_output_dirs(config):
    # Extract the video name from the input video path
    video_name = os.path.splitext(os.path.basename(config['input_video_path']))[0]
    
    # Create the path for the output video directory
    output_video_dir = os.path.join(config['output_base_dir'], video_name)
    config['output_video_dir'] = output_video_dir
    
    # Check if the directory exists, if not create it
    if not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir)
    
    # Set the path for the output CSV file
    config['output_csv_path'] = os.path.join(output_video_dir, 'video_detections.csv')
    
    # Set the path for the output video file
    config['output_video_path'] = os.path.join(output_video_dir, 'processed_grid_video.mp4')
    
    # Set the path for the HSV ranges text file
    config['hsv_ranges_path'] = os.path.join(output_video_dir, 'hsv_ranges.txt')
    
    return config


if __name__ == "__main__":
    config = {
        'input_video_path': '../../../Datasets/soccer_field_homography/video_test_1.mp4',
        'input_layout_image': '../../../Datasets/soccer field layout/soccer_field_layout.png',
        'input_layout_array': '../../../Datasets/soccer field layout/soccer_field_layout_points.npy',
        'yolo_model_path': '../../../Models/pretrained-yolov8-soccer.pt',
        'output_base_dir': '../outputs'
    }
    
    config = create_output_dirs(config)

    classes_hsv_ranges = setup_hsv_ranges(config, n_classes=2)
    processor = VideoProcessor(config, classes_hsv_ranges)
    processor.process_video()