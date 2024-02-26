import os
import numpy as np
from ultralytics import YOLO
import cv2
from tqdm import tqdm
from norfair import Detection, Tracker

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
        
    config['output_video_path'] = os.path.join(output_video_dir, 'output_video.mp4')
    
    return config

# Object detection functions --------------------------------------------------
class DetectedObject:
    def __init__(self, bbox, lbl):
        self.bbox = bbox
        self.label = lbl
    def get_bbox_bottom(self):
        xmin, ymin, xmax, ymax = self.bbox
        return np.array([int((xmin + xmax) / 2), int(ymax)])
        
class DetectionProcessor:
    def __init__(self, config):
        self.model = YOLO(config["yolo_model_path"])
        
        # Tracker variables
        self.distance_function = "euclidean"
        self.distance_threshold = int(config["distance_threshold_centroid"])
        self.max_distance = int(config["max_distance"])
        
        self.tracker = Tracker(distance_function=self.distance_function, 
                               distance_threshold=self.distance_threshold)

    def compute_detected_objects(self, yolo_detections, frame):
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
    
    def convert_yolo_to_norfair(self, yolo_detections, label):
        norfair_detections = []

        detections_as_xywh = yolo_detections[0].boxes.xywh
        scores = yolo_detections[0].boxes.conf
        labels = yolo_detections[0].boxes.cls
        
        for det_xywh, scr, lbl in zip(detections_as_xywh, scores, labels):
            lbl = int(lbl.cpu().numpy())
            
            if lbl == label:
                det_xywh = det_xywh.cpu().numpy()
                x, y, w, h = det_xywh
                
                centroid = np.array([x, y])
                
                scr = np.array([float(scr.cpu().numpy())])
                
                norfair_detections.append(
                    Detection(
                        points=centroid,
                        scores=scr,
                        label=int(lbl),
                    )
                )
        return norfair_detections

# Specific objects classes ----------------------------------------------------
class Ball:
    def __init__(self, yolo_label):
        self.yolo_label = yolo_label
        self.color = (255, 255, 255)
        self.last_team_label = None
        self.center_point = None

    def update_draw_location(self, players_obj, detected_objects, frame):
        detected_bbox = [obj.bbox for obj in detected_objects if obj.label == self.yolo_label]
        
        if len(detected_bbox) > 0:
            bbox_areas = np.array([(box[2] - box[0]) * (box[3] - box[1]) for box in detected_bbox])
            
            index_max_area = np.argmax(bbox_areas)
            max_bbox = detected_bbox[index_max_area]
            center_x = int((max_bbox[0] + max_bbox[2]) / 2)
            center_y = int((max_bbox[1] + max_bbox[3]) / 2)
            
            self.center_point = (center_x, center_y)
            
            for player in players_obj:
                if player.max_bbox is not None:
                    px1, py1, px2, py2 = player.max_bbox
                    
                    if (center_x > px1) and (center_x < px2):
                        if (center_y > py1 + 2*py2/3) and (center_y < py2):
                            self.color = player.color
                            self.last_team_label = player.yolo_label
                    
            self.draw_object(max_bbox, frame)
        else:
            self.center_point = None
            
    def draw_object(self, bbox, frame):
        color = self.color
        
        # Calculate the center and size of the bounding box
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
    
        # Determine the radius as half the smaller of the width or height
        radius = int(min(width, height) / 2)
    
        # Draw the circle
        cv2.circle(frame, (center_x, center_y), radius, color, -1)
    
class Player:
    def __init__(self, yolo_label, team_name):
        team_colors = {2: (255, 0, 0), 
                       3: (0, 0, 255)}
        
        self.yolo_label = yolo_label
        self.team_name = team_name
        self.color = team_colors[self.yolo_label]
        self.max_bbox = None
        
    def update_draw_location(self, detected_objects, frame):
        detected_bbox = [obj.bbox for obj in detected_objects if obj.label == self.yolo_label]
        
        if len(detected_bbox) > 0:
            bbox_areas = np.array([(box[2] - box[0]) * (box[3] - box[1]) for box in detected_bbox])
            
            index_max_area = np.argmax(bbox_areas)
            self.max_bbox = detected_bbox[index_max_area]
            
            self.draw_object(self.max_bbox, frame)
        else:
            self.max_bbox = None

    def draw_object(self, bbox, frame):
        color = self.color
        text = self.team_name
        
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        # Define colors and font for the bounding box and label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 2

        # Draw the bounding box
        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

        # Calculate text size for background
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # Draw background for text for better visibility
        frame = cv2.rectangle(frame, (xmin, ymin - 20), (xmin + text_width, ymin), color, -1)

        # Put the text (label) on the frame
        frame = cv2.putText(frame, text, (xmin, ymin - 5), font, font_scale, (255, 255, 255), font_thickness)

class GoalPolygon:
    def __init__(self, config):
        self.config = config
        self.polygon = []
        self.load_and_draw_polygon()
        self.ball_in = False
        self.teams_scores = [0, 0]
        self.team_label_idx = {2: 0,
                               3: 1}
        
    def load_polygon_if_exists(self):
        polygon_path = os.path.join(self.config['output_video_dir'], 'polygon.npy')
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
        cv2.imshow("First Frame - Draw Polygon (Press 'q' to proceed)", self.first_frame)

    def load_and_draw_polygon(self):
        cap = cv2.VideoCapture(self.config['input_video_path'])
        ret, self.first_frame = cap.read()
        if not ret:
            print("Error loading the first frame.")
            return
        cap.release()

        if self.load_polygon_if_exists():
            # Draw the loaded polygon on the first frame for visualization
            if len(self.polygon) > 1:
                cv2.polylines(self.first_frame, [np.array(self.polygon)], True, (0, 255, 0), 2)
        else:
            cv2.namedWindow("First Frame - Draw Polygon (Press 'q' to proceed)")
            cv2.setMouseCallback("First Frame - Draw Polygon (Press 'q' to proceed)", self.draw_polygon)

            while True:
                cv2.imshow("First Frame - Draw Polygon (Press 'q' to proceed)", self.first_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    if len(self.polygon) > 2 and self.polygon[0] != self.polygon[-1]:
                        self.polygon.append(self.polygon[0])  # Close the polygon if necessary
                    self.save_polygon()  # Save the newly drawn polygon
                    break
            cv2.destroyAllWindows()

    def save_polygon(self):
        polygon_path = os.path.join(self.config['output_video_dir'], 'polygon.npy')
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
    
        
    
    
# General video processing functions ------------------------------------------
class VideoProcessor:
    def __init__(self, config):
        self.config = config
        self.detection_processor = DetectionProcessor(config)
        self.first_frame = None
        
        self.goal_polygon = GoalPolygon(config)
        self.ball_object = Ball(1)
        self.player_a = Player(2, "Team A")
        self.player_b = Player(3, "Team B")
        
    def process_frame(self, frame):
        yolo_detections = self.detection_processor.model.predict(frame, 
                                                                 verbose=False, 
                                                                 conf=0.5, 
                                                                 imgsz=640)
        
        # norfair_ball_detections = self.detection_processor.convert_yolo_to_norfair(yolo_detections, label=1)
        # tracked_balls = self.detection_processor.tracker.update(detections=norfair_ball_detections)
        
        # for obj in tracked_balls:
        #     print(obj.estimate)
        self.goal_polygon.draw_polygon_on_frame(frame)
        
        detected_objects = self.detection_processor.compute_detected_objects(yolo_detections, frame)
        
        self.player_a.update_draw_location(detected_objects, frame)
        self.player_b.update_draw_location(detected_objects, frame)
        
        self.ball_object.update_draw_location([self.player_a, self.player_b], detected_objects, frame)
        
        self.goal_polygon.update_draw_score(self.ball_object, frame)

        return frame
        
    def process_video(self):
        cap = cv2.VideoCapture(self.config['input_video_path'])
    
        # Check if video opened successfully
        if not cap.isOpened():
            print("Error opening video stream or file")
            return
        
        # Get video frame width, height, and FPS for the output video
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4 files
        out = cv2.VideoWriter(self.config['output_video_path'], fourcc, fps, (frame_width, frame_height))
    
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with tqdm(total=total_frames, desc="Processing video frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    processed_frame = self.process_frame(frame)
                    out.write(processed_frame)
    
                    # Display the single composite window
                    cv2.imshow("Detections", processed_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    
                    pbar.update(1)
                else:
                    break

	# Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()


# Main function ---------------------------------------------------------------
if __name__ == "__main__":
    config = {
        'input_video_path': '../../../Datasets/demo/demo_v2_sliced.mp4',
        'yolo_model_path': '../../../Models/yolov8-demo-model/train/weights/best.pt',
        'output_base_dir': '../outputs_v2',
        'distance_threshold_centroid': 30, 
        'max_distance': 10000
    }
    
    config = create_output_dirs(config)
    processor = VideoProcessor(config)
    processor.process_video()
