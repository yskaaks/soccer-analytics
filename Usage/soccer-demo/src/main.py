import os
import json
import numpy as np
import cv2
from ultralytics import YOLO
import random
import math
import sys
from tqdm import tqdm

# Utility functions ===========================================================
def create_output_dirs(config):
    # Extract the video name from the input video path
    video_name = os.path.splitext(os.path.basename(config['input_video_path']))[0]
    
    # Create the path for the output video directory
    output_video_dir = os.path.join(config['output_base_dir'], video_name)
    config['output_video_dir'] = output_video_dir

    # Check if the directory exists, if not create it
    if not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir)
    
    # Define paths for input data
    config['hsv_ranges_path'] = os.path.join(output_video_dir, 'hsv_ranges.txt')
    config['h_matrix_path'] = os.path.join(output_video_dir, 'h_matrix.npy')
    config['goal_polygon_path'] = os.path.join(output_video_dir, 'goal_polygon.npy')
    
    # Define paths for output data
    config['output_video_path'] = os.path.join(output_video_dir, 'scores_video.mp4')
    config['output_video_heatmap_a'] = os.path.join(output_video_dir, 'heatmap_a_video.mp4')
    config['output_video_heatmap_b'] = os.path.join(output_video_dir, 'heatmap_b_video.mp4')
    config['output_heatmap_a'] = os.path.join(output_video_dir, 'heatmap_a.png')
    config['output_heatmap_b'] = os.path.join(output_video_dir, 'heatmap_b.png')
    config['output_csv'] = os.path.join(output_video_dir, 'soccer_analytics.csv')
    
    return config

# Setup stage =================================================================
# HSV ranges per class --------------------------------------------------------
class HSVRangeSetup:
    def __init__(self, config, n_classes=2, labels_of_interest=[2, 3]):
        self.config = config
        self.n_classes = n_classes
        self.labels_of_interest = labels_of_interest
        
    def load_hsv_ranges(self, file_path):
        try:
            with open(file_path, 'r') as file:
                hsv_ranges_list = json.load(file)
            hsv_ranges = [(np.array(lower), np.array(upper)) for lower, upper in hsv_ranges_list]
            return hsv_ranges
        except Exception as e:
            raise ValueError(f"Error loading HSV ranges: {e}")

    def save_hsv_ranges(self, hsv_ranges, file_path):
        try:
            hsv_ranges_list = [(lower.tolist(), upper.tolist()) for lower, upper in hsv_ranges]
            with open(file_path, 'w') as file:
                json.dump(hsv_ranges_list, file)
        except Exception as e:
            raise ValueError(f"Error saving HSV ranges: {e}")

    def extract_and_detect_frames(self, video_path, object_detector, labels_of_interest):
        """
        Randomly selects and processes 5 frames from the video, displaying and collecting crops
        from detections for specified labels of interest.
        """
        detected_crops = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Error: Could not open video.")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            random_frames = sorted(random.sample(range(total_frames), 5))

            for frame_index in tqdm(random_frames, desc="Detecting players"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if not ret:
                    continue  # Skip if the frame could not be read

                # Detect objects in the frame
                detected_objects = object_detector.detect(frame)
                for det in detected_objects:
                    if det.label in labels_of_interest:
                        xmin, ymin, xmax, ymax = [int(x) for x in det.bbox]
                        crop = frame[ymin:ymax, xmin:xmax]
                        detected_crops.append(crop)
        finally:
            cap.release()

        return detected_crops

    # Function to resize and concatenate crops in a grid format
    def scale_and_concat_crops(self, crops, max_crops_per_row=10, max_crops_per_col=5, display_height=300, display_width=800):
        # Calculate the number of rows needed for the grid
        num_rows = min(math.ceil(len(crops) / max_crops_per_row), max_crops_per_col)
    
        # Initialize a blank image to hold all crops
        grid_image = np.zeros((display_height, display_width, 3), dtype=np.uint8)
    
        # Calculate the size for each crop
        crop_height = display_height // num_rows
        crop_width = display_width // max_crops_per_row
    
        # Store the starting y-coordinate for the current row
        current_y = 0
    
        for i, crop in enumerate(crops):
            # Resize crop to fit within the grid cell
            resized_crop = cv2.resize(crop, (crop_width, crop_height))
    
            # Calculate the row and column for this crop
            row = i // max_crops_per_row
            col = i % max_crops_per_row
    
            # Calculate the starting x-coordinate for this crop
            current_x = col * crop_width
            current_y = row * crop_height
    
            # Place the resized crop into the grid image
            end_x = current_x + resized_crop.shape[1]
            end_y = current_y + resized_crop.shape[0]
            
            # Check if the crop can be placed without exceeding the grid's bounds
            if end_y <= grid_image.shape[0] and end_x <= grid_image.shape[1]:
                grid_image[current_y:end_y, current_x:end_x] = resized_crop
            else:
                # If the bounds are exceeded, break the loop
                break
    
        return grid_image

    def setup_hsv_ranges(self, object_detector):
        """
        Sets up HSV ranges using detected objects from an object detector model.
        Prints instructions in the console and only allows 'y' to advance or 'esc' to quit.
        """
        if os.path.exists(self.config.get('hsv_ranges_path', '')):
            return self.load_hsv_ranges(self.config['hsv_ranges_path'])
    
        detected_crops = self.extract_and_detect_frames(self.config['input_video_path'],
                                                        object_detector,
                                                        self.labels_of_interest)
    
        concatenated_crops = self.scale_and_concat_crops(detected_crops)
        segmentation = VideoSegmentation('Segmentation')
        hsv_ranges = []
        
        print("Instructions:")
        print("- Press 'y' if you are satisfied with the HSV range and want to move to the next class.")
        print("- Press 'Esc' to exit the program.")
    
        for _ in range(self.n_classes):
            while True:
                lower_bound, upper_bound = segmentation.update_segmentation(concatenated_crops)
                key = cv2.waitKey(1) & 0xFF
    
                if key == ord('y'):
                    hsv_ranges.append((lower_bound, upper_bound))
                    break
                elif key == 27:  # Escape key
                    cv2.destroyAllWindows()
                    sys.exit()
    
            segmentation.reset_trackbars()
    
        cv2.destroyAllWindows()
        self.save_hsv_ranges(hsv_ranges, self.config['hsv_ranges_path'])
        return hsv_ranges

class VideoSegmentation:
    def __init__(self, window_name):
        self.window_name = window_name
        cv2.namedWindow(self.window_name)
        self.lower_bound = np.array([0, 0, 0])
        self.upper_bound = np.array([179, 255, 255])
        self.create_trackbars()

    def create_trackbars(self):
        cv2.createTrackbar('H Min', self.window_name, 0, 179, self.noop)
        cv2.createTrackbar('H Max', self.window_name, 179, 179, self.noop)
        cv2.createTrackbar('S Min', self.window_name, 0, 255, self.noop)
        cv2.createTrackbar('S Max', self.window_name, 255, 255, self.noop)
        cv2.createTrackbar('V Min', self.window_name, 0, 255, self.noop)
        cv2.createTrackbar('V Max', self.window_name, 255, 255, self.noop)

    def noop(self, x):
        pass

    def get_hsv_ranges(self):
        return self.lower_bound, self.upper_bound

    def update_segmentation(self, concatenated_frame):
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
        cv2.setTrackbarPos('H Min', self.window_name, 0)
        cv2.setTrackbarPos('H Max', self.window_name, 179)
        cv2.setTrackbarPos('S Min', self.window_name, 0)
        cv2.setTrackbarPos('S Max', self.window_name, 255)
        cv2.setTrackbarPos('V Min', self.window_name, 0)
        cv2.setTrackbarPos('V Max', self.window_name, 255)
        
        
# Manual homography matrix ----------------------------------------------------
class HomographySetup:
    def __init__(self, config):
        self.config = config
        self.layout_img, self.first_frame = self.load_and_prepare_images()
        self.points_layout = []
        self.points_frame = []

    def load_and_prepare_images(self):
        layout_img = cv2.imread(self.config['input_layout_image'])
        cap = cv2.VideoCapture(self.config['input_video_path'])
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Error: Could not read the video frame.")
            return None, None

        return layout_img, frame

    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            concatenated_img, max_width = params
            if x < max_width:  # Clicked on the layout image
                self.points_layout.append((x, y))
            else:  # Clicked on the video frame
                self.points_frame.append((x - max_width, y))
            self.update_display(concatenated_img, max_width)

    def update_display(self, concatenated_img, max_width):
        concatenated_img[:, :] = np.concatenate((self.padded_layout_img, self.padded_first_frame), axis=1)
        for pt_layout in self.points_layout:
            cv2.circle(concatenated_img, pt_layout, 5, (255, 0, 0), -1)
        for pt_frame in self.points_frame:
            cv2.circle(concatenated_img, (pt_frame[0] + max_width, pt_frame[1]), 5, (0, 255, 0), -1)
        for pt_layout, pt_frame in zip(self.points_layout, self.points_frame):
            cv2.line(concatenated_img, pt_layout, (pt_frame[0] + max_width, pt_frame[1]), (0, 0, 255), 2)
        cv2.imshow("Homography Points Selection", concatenated_img)

    def compute_homography_matrix(self):
        if self.config['h_matrix_path'] and os.path.exists(self.config['h_matrix_path']):
            return np.load(self.config['h_matrix_path'])

        self.padded_layout_img, self.padded_first_frame, concatenated_img = self.prepare_images_for_display()
        max_width = max(self.layout_img.shape[1], self.first_frame.shape[1])

        cv2.namedWindow("Homography Points Selection", cv2.WINDOW_NORMAL)
        cv2.imshow("Homography Points Selection", concatenated_img)
        cv2.setMouseCallback("Homography Points Selection", self.click_event, (concatenated_img, max_width))

        print("Instructions:")
        print("- Click corresponding points on the layout image and the video frame.")
        print("- Press 'y' to confirm and calculate homography.")
        print("- Press 'Esc' to quit.")
        print("- Press 'r' to remove the last point match.")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Esc key
                cv2.destroyAllWindows()
                return None  # Return early since homography cannot be computed
            elif key == ord('y'):
                if len(self.points_layout) >= 4 and len(self.points_frame) >= 4:
                    H, _ = cv2.findHomography(np.array(self.points_frame), np.array(self.points_layout))
                    if self.config['h_matrix_path']:
                        np.save(self.config['h_matrix_path'], H)
                    cv2.destroyAllWindows()
                    return H
                else:
                    print("Not enough points to compute homography.")
                    cv2.destroyAllWindows()
                    return None  # Return early since homography cannot be computed
            elif key == ord('r') and self.points_layout and self.points_frame:  # Remove the last point match
                self.points_layout.pop()
                self.points_frame.pop()
                self.update_display(concatenated_img, max_width)

    def prepare_images_for_display(self):
        max_height = max(self.layout_img.shape[0], self.first_frame.shape[0])
        max_width = max(self.layout_img.shape[1], self.first_frame.shape[1])

        padded_layout_img = cv2.copyMakeBorder(self.layout_img, 0, max_height - self.layout_img.shape[0], 0, max_width - self.layout_img.shape[1], cv2.BORDER_CONSTANT, value=[0, 0, 0])
        padded_first_frame = cv2.copyMakeBorder(self.first_frame, 0, max_height - self.first_frame.shape[0], 0, max_width - self.first_frame.shape[1], cv2.BORDER_CONSTANT, value=[0, 0, 0])

        concatenated_img = np.concatenate((padded_layout_img, padded_first_frame), axis=1)
        return padded_layout_img, padded_first_frame, concatenated_img
    
# Manual goal polygon ---------------------------------------------------------
class GoalPolygon:
    def __init__(self, config):
        self.config = config
        self.first_frame = self.load_first_frame()
        self.polygon = []
        self.load_and_draw_polygon()
        self.ball_in = False
        self.teams_scores = [0, 0]
        self.team_label_idx = {2: 0,
                               3: 1}
        
    def load_polygon_if_exists(self):
        polygon_path = self.config['goal_polygon_path']
        if os.path.exists(polygon_path):
            self.polygon = np.load(polygon_path)
            return True
        return False

    def draw_polygon(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.polygon.append((x, y))
            cv2.circle(self.first_frame, (x, y), 5, (0, 255, 0), -1)
        if len(self.polygon) > 1:
            cv2.polylines(self.first_frame, [np.array(self.polygon)], False, (255, 0, 0), 2)
        cv2.imshow("Goal Polygon", self.first_frame)
    
    def load_first_frame(self):
        cap = cv2.VideoCapture(self.config['input_video_path'])
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Error loading the first frame.")
            return None
        return frame
    
    def load_and_draw_polygon(self):
        if not self.load_polygon_if_exists():
            cv2.namedWindow("Goal Polygon")
            cv2.setMouseCallback("Goal Polygon", self.draw_polygon)
    
            print("Instructions:")
            print("- Draw the goal polygon. Click to add points.")
            print("- Press 'y' to save and continue.")
            print("- Press 'Esc' to exit without saving.")
            print("- Press 'r' to remove the last point.")
    
            while True:
                cv2.imshow("Goal Polygon", self.first_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('y'):
                    # Close the polygon if it's not already closed
                    if len(self.polygon) > 2 and self.polygon[0] != self.polygon[-1]:
                        self.polygon.append(self.polygon[0])
                    self.save_polygon()
                    break
                elif key == 27:  # Esc key
                    print("Exiting without saving the polygon.")
                    cv2.destroyAllWindows()
                elif key == ord('r') and self.polygon:  # Remove the last point
                    self.polygon.pop()
                    self.first_frame = self.load_first_frame()  # Reload the frame to clear previous drawings
                    if len(self.polygon) > 1:
                        cv2.polylines(self.first_frame, [np.array(self.polygon)], False, (0, 255, 0), 2)
                    for point in self.polygon:
                        cv2.circle(self.first_frame, point, 5, (0, 255, 0), -1)
    
            cv2.destroyAllWindows()

    def save_polygon(self):
        polygon_path = self.config['goal_polygon_path']
        np.save(polygon_path, np.array(self.polygon))
        
    def draw_polygon_on_frame(self, frame):
        # Check if polygon has points
        if len(self.polygon) > 0:
            # Convert polygon points to a numpy array with shape (n, 1, 2) for cv2.polylines compatibility
            pts = np.array(self.polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))
            # Draw the polygon on the frame
            
            if self.ball_in:
                cv2.fillPoly(frame, [pts], color=(0, 255, 0))
            else:
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
            
    def update_draw_score(self, ball_object, frame):
        ball_center = ball_object.center_point
        
        if ball_center is not None:
            pts = np.array(self.polygon, np.int32)
            result = int(cv2.pointPolygonTest(pts, ball_center, False))
            
            if not self.ball_in and result == 1:
                self.ball_in = True
                
                team_label = ball_object.last_team_label
                
                if team_label is not None:
                    team_idx = self.team_label_idx[team_label]
                    self.teams_scores[team_idx] += 1
            if self.ball_in and result == -1:
                self.ball_in = False
            
        self.draw_score_box(frame)
            
    def draw_score_box(self, frame):
        scores = self.teams_scores
        
        # Frame dimensions
        height, width = frame.shape[:2]
    
        # Score box dimensions and position
        box_width = 200
        box_height = 50
        top_left_x = width // 2 - box_width // 2
        top_left_y = 20
        bottom_right_x = top_left_x + box_width
        bottom_right_y = top_left_y + box_height
    
        # Label box dimensions and position (directly above the score box)
        label_box_height = 20
        label_top_left_y = top_left_y - label_box_height
        label_bottom_right_y = top_left_y
    
        # Draw the label box
        cv2.rectangle(frame, (top_left_x, label_top_left_y), (bottom_right_x, label_bottom_right_y), (0, 0, 0), -1)
    
        # Draw the white score box
        cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (255, 255, 255), -1)
    
        # Prepare and draw the "Points" label in the label box
        label_text = "(A) - Points - (B)"
        label_font_scale = 0.5
        label_font_thickness = 1
        label_text_color = (255, 255, 255)  # White text
        label_text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_font_thickness)[0]
        label_text_x = top_left_x + (box_width - label_text_size[0]) // 2
        label_text_y = label_top_left_y + (label_box_height + label_text_size[1]) // 2
        cv2.putText(frame, label_text, (label_text_x, label_text_y), cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_text_color, label_font_thickness)
    
        # Prepare and draw the score text in the score box
        score_text = f"{scores[0]} - {scores[1]}"
        score_font_scale = 1
        score_font_thickness = 2
        score_text_color = (0, 0, 0)  # Black text
        score_text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, score_font_scale, score_font_thickness)[0]
        score_text_x = top_left_x + (box_width - score_text_size[0]) // 2
        score_text_y = top_left_y + (box_height + score_text_size[1]) // 2
        cv2.putText(frame, score_text, (score_text_x, score_text_y), cv2.FONT_HERSHEY_SIMPLEX, score_font_scale, score_text_color, score_font_thickness)
    


# Object detection functions ==================================================
class DetectedObject:
    def __init__(self, bbox, lbl):
        self.bbox = [int(j) for j in bbox]
        self.label = lbl
    def get_bbox_bottom(self):
        xmin, ymin, xmax, ymax = self.bbox
        return np.array([int((xmin + xmax) / 2), int(ymax)])
        
class ObjectDetector:
    def __init__(self, config):
        self.model = YOLO(config["yolo_model_path"])
        
    def detect(self, frame, verbose=False, conf=0.5, imgsz=640):
        yolo_detections = self.model.predict(frame, verbose=verbose, conf=conf, imgsz=imgsz)
        detected_objects = self.compute_detected_objects(yolo_detections)
        
        return detected_objects

    def compute_detected_objects(self, yolo_detections):
        detected_objects = []
        detections_as_xyxy = yolo_detections[0].boxes.xyxy
        labels = yolo_detections[0].boxes.cls

        for lbl, det_xyxy in zip(labels, detections_as_xyxy):
            det_xyxy = det_xyxy.cpu().numpy()
            lbl = int(lbl.cpu().numpy())

            x1, y1, x2, y2 = det_xyxy
            det_object = DetectedObject(det_xyxy, lbl)
            detected_objects.append(det_object)
        
        return detected_objects


# Main function ===============================================================
if __name__ == "__main__":
    # 1. Setup
    config = {
        'input_video_path': '../../../Datasets/demo/demo_v2_sliced.mp4',
        'input_layout_image': '../../../Datasets/soccer field layout/soccer_field_layout.png',
        'yolo_model_path': '../../../Models/yolov8-demo-model/train/weights/best.pt',
        'output_base_dir': '../outputs'
    }
    
    config = create_output_dirs(config)
    
    object_detector = ObjectDetector(config)
    
    # HSV Ranges setup
    hsv_setup = HSVRangeSetup(config, n_classes=2, labels_of_interest=[2, 3])
    hsv_ranges = hsv_setup.setup_hsv_ranges(object_detector)
    
    # H Matrix setup
    homography_setup = HomographySetup(config)
    H = homography_setup.compute_homography_matrix()
    
    # Goal polygon setup
    goal_polygon = GoalPolygon(config)
    
    
    
    
    












