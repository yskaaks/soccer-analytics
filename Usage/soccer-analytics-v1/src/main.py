
import cv2
import numpy as np
import os
import logging
import pandas as pd
from typing import Dict, Any, Tuple
from tqdm import tqdm
from ultralytics import YOLO
from utils import homography_transformation_process

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HomographyState:
    def __init__(self, guess_fx: float, guess_rot: np.ndarray, guess_trans: Tuple[float, float, float]):
        self.guess_fx = guess_fx
        self.guess_rot = guess_rot
        self.guess_trans = guess_trans

    def update_state(self, guess_fx: float, guess_rot: np.ndarray, guess_trans: Tuple[float, float, float]):
        self.guess_fx = guess_fx
        self.guess_rot = guess_rot
        self.guess_trans = guess_trans

class DetectedObject:
    def __init__(self, bbox, obj_id):
        self.bbox = bbox
        self.id = obj_id

def compute_detected_objects(yolo_detections, frame):
    detected_objects = []
    detections_as_xyxy = yolo_detections[0].boxes.xyxy
    detections_ids = [0 for _ in detections_as_xyxy]
    
    for det_xyxy, det_id in zip(detections_as_xyxy, detections_ids):
        det_xyxy = det_xyxy.cpu().numpy()
        det_object = DetectedObject(det_xyxy, det_id)
        detected_objects.append(det_object)
    
    return detected_objects

def draw_detected_objects(original_frame, detected_objects, circle_color):
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
        # box_id = 0
        
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

class VideoProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state = HomographyState(guess_fx=2000, guess_rot=np.array([[0.25, 0, 0]]), guess_trans=(0, 0, 80))
        self.template_img_rgb = self.load_template_image(config['input_layout_image'])
        self.key_points_layout = np.load(config['input_layout_array'])
        self.model = YOLO(config["yolo_model_path"])
        # self.ocr_reader = easyocr.Reader(['en'], gpu=config["use_gpu"])
        video_name = os.path.splitext(os.path.basename(config["input_video_path"]))[0]
        self.output_dir = os.path.join(config["output_base_dir"], video_name, "2d_projections")
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_video_path = os.path.join(config["output_base_dir"], video_name, video_name + "_processed.mp4")
        self.grid_video_path = os.path.join(config["output_base_dir"], video_name, video_name + "_grid.mp4")
        
        self.csv_path = os.path.join(self.config["output_base_dir"], video_name, "detection_details.csv")
        self.initialize_csv()
        
    def initialize_csv(self):
        # Check if the CSV file exists and delete it if it does
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        # Create an empty DataFrame with the appropriate columns and save it as a CSV file
        pd.DataFrame(columns=['frame', 'H matrix', 'id', 'bbox center point coordinates in original frame', 'transformed bbox center point with H matrix']).to_csv(self.csv_path, index=False)

    def append_detection_details_to_csv(self, details):
        # Convert the single detection detail into a DataFrame and append it to the CSV file
        df = pd.DataFrame([details])
        df.to_csv(self.csv_path, mode='a', header=False, index=False)
        
    @staticmethod
    def load_template_image(path: str) -> np.ndarray:
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

    def process_frame(self, frame: np.ndarray, frame_number: int):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            H, _, guess_fx, guess_rot, guess_trans = homography_transformation_process(
                frame_rgb, self.template_img_rgb, self.key_points_layout, 
                self.state.guess_fx, self.state.guess_rot, self.state.guess_trans)
            
            detections = self.model.track(frame_rgb, persist=True, verbose=False)

            if H is not None:
                self.state.update_state(guess_fx, guess_rot, guess_trans)
                detected_objects = compute_detected_objects(detections, frame_rgb)
                
                for det_object in detected_objects:
                    bbox_center = self.get_bbox_center(det_object.bbox)
                    transformed_center = self.convert_to_homography_point(bbox_center, H)
                    details = {
                        'frame': frame_number,
                        'H matrix': H.flatten().tolist(),
                        'id': det_object.id,
                        'bbox center point coordinates in original frame': bbox_center,
                        'transformed bbox center point with H matrix': transformed_center
                    }
                    self.append_detection_details_to_csv(details)

                # Reset the template image for each frame to avoid drawing over previous points
                self.template_img_rgb = self.load_template_image(self.config['input_layout_image'])
                
                for det_object in detected_objects:
                    bbox_center = self.get_bbox_center(det_object.bbox)
                    point_2d = self.convert_to_homography_point(bbox_center, H)
                    self.draw_point_on_template(point_2d)
                self.save_template_image(frame_number)

                # Draw detections using the provided function
                frame_with_detections = draw_detected_objects(frame_rgb, detected_objects, (255, 0, 0))
                return frame_with_detections, frame_rgb, self.template_img_rgb, H  # Return H here
            else:
                return None, None, None, None
        except Exception as e:
            logging.error(f"Error processing frame {frame_number}: {e}")
            return None, None, None, None
        
    def get_bbox_center(self, bbox) -> Tuple[int, int]:
        xmin, ymin, xmax, ymax = bbox
        return int((xmin + xmax) / 2), int(ymax)

    def convert_to_homography_point(self, point: Tuple[int, int], H: np.ndarray) -> Tuple[int, int]:
        point_homogeneous = np.array([*point, 1]).reshape(-1, 1)
        transformed_point = np.dot(H, point_homogeneous)
        transformed_point /= transformed_point[2]

        return int(transformed_point[0]), int(transformed_point[1])

    def draw_point_on_template(self, point: Tuple[int, int]):
        cv2.circle(self.template_img_rgb, point, radius=5, color=(0, 0, 255), thickness=-1)
        cv2.circle(self.template_img_rgb, point, radius=6, color=(0, 0, 0), thickness=1)

    def save_template_image(self, frame_number: int):
        output_path = os.path.join(self.output_dir, f"frame_{frame_number:04d}.png")
        cv2.imwrite(output_path, cv2.cvtColor(self.template_img_rgb, cv2.COLOR_RGB2BGR))
        
    def resize_images_to_smallest(self, *images):
        min_height = min(image.shape[0] for image in images)
        min_width = min(image.shape[1] for image in images)
        resized_images = [cv2.resize(image, (min_width, min_height)) for image in images]
        return resized_images
    
    def compose_grid(self, images, rows=2, cols=2):
        img_height, img_width = images[0].shape[:2]
        grid = np.zeros((img_height * rows, img_width * cols, 3), dtype=images[0].dtype)
        for idx, image in enumerate(images):
            row, col = divmod(idx, cols)
            grid[row*img_height:(row+1)*img_height, col*img_width:(col+1)*img_width, :] = image
        return grid


    def process_video(self):
        cap = cv2.VideoCapture(self.config['input_video_path'])
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_video_size = (int(cap.get(3)), int(cap.get(4)))
        grid_video_size = None  # To be determined after processing the first frame
    
        # Initialize the original video writer
        original_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        original_out = cv2.VideoWriter(self.output_video_path, original_fourcc, 20.0, original_video_size)
    
        # Placeholder for grid video writer initialization
        grid_out = None
    
        with tqdm(total=total_frames, desc="Processing video frames") as pbar:
            frame_number = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    processed_frame, original_frame_rgb, template_img_rgb, H = self.process_frame(frame, frame_number)
                    
                    if processed_frame is not None and H is not None:
                        # Compute the transformed original frame using H
                        transformed_frame = cv2.warpPerspective(original_frame_rgb, H, (template_img_rgb.shape[1], template_img_rgb.shape[0]))
                        
                        # Assuming process_frame also properly handles and returns the template_img_rgb for the current frame
                        final_points_frame = template_img_rgb
                        
                        # Resize images to the smallest size among them for uniformity
                        # resized_images = self.resize_images_to_smallest(original_frame_rgb, processed_frame, transformed_frame, final_points_frame)
                        resized_images = self.resize_images_to_smallest(cv2.cvtColor(original_frame_rgb, cv2.COLOR_RGB2BGR), 
                                                                        cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR), 
                                                                        cv2.cvtColor(transformed_frame, cv2.COLOR_RGB2BGR), 
                                                                        final_points_frame)
                        
                        # Create a 2x2 grid
                        grid_frame = self.compose_grid(resized_images)
                        
                        # Initialize or update the grid video writer based on the grid frame size
                        if grid_out is None:
                            grid_video_size = (grid_frame.shape[1], grid_frame.shape[0])
                            grid_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            grid_out = cv2.VideoWriter(self.grid_video_path, grid_fourcc, 20.0, grid_video_size)
                            
                            
                        cv2.imshow("layout", final_points_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                            
                        # Write the grid frame to the grid output video
                        grid_out.write(grid_frame)
                    
                    # Write the processed frame to the original output video if available
                    if processed_frame is not None:
                        original_out.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
    
                    frame_number += 1
                    pbar.update(1)
                else:
                    break
    
        # Release all resources
        cap.release()
        original_out.release()
        if grid_out is not None:
            grid_out.release()
            
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    config = {
        'input_video_path': '../../../Datasets/soccer_field_homography/video_test_2.mp4',
        'input_layout_image': '../../../Datasets/soccer field layout/soccer_field_layout.png',
        'input_layout_array': '../../../Datasets/soccer field layout/soccer_field_layout_points.npy',
        'yolo_model_path': '../../../Models/pretrained-yolov8-soccer.pt',
        'output_base_dir': '../outputs',
        'use_gpu': True
    }
    processor = VideoProcessor(config)
    processor.process_video()

