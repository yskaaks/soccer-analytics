from utils import (create_progress_bar, 
                   find_continuous_sequences, 
                   update_sequences_to_uniform_size,
                   update_tracks_csv)
from opencv_process import (create_video_writer, 
                            draw_tracked_objects, 
                            save_sequences_as_videos)
from norfair_process import compute_tracked_objects
from ultralytics import YOLO
import cv2
from norfair import Tracker
from norfair.camera_motion import HomographyTransformationGetter, MotionEstimator
import contextlib
import os

# Configuration
CONFIG = {
    "yolo_model": "../../../Models/pretrained-yolov8-soccer.pt",
    "video_path": "../../../Datasets/soccer_field_homography/video_test_1.mp4",
    "output_base_dir": "../outputs"  # Base directory for all outputs
}

@contextlib.contextmanager
def video_capture(path):
    cap = cv2.VideoCapture(path)
    try:
        yield cap
    finally:
        cap.release()

@contextlib.contextmanager
def video_writer(*args, **kwargs):
    writer = create_video_writer(*args, **kwargs)
    try:
        yield writer
    finally:
        writer.release()
        

def process_frame(frame, model, tracker, motion_estimator, prev_tracked_objects):
    results = model(frame, classes=[1, 3], verbose=False)
    tracked_objects = compute_tracked_objects(results, prev_tracked_objects, 
                                              frame, motion_estimator, tracker)
    
    return draw_tracked_objects(frame, tracked_objects, circle_color=(255, 0, 0)), tracked_objects
    

def main(config):
    base_name = os.path.splitext(os.path.basename(config["video_path"]))[0]
    output_dir = os.path.join(config["output_base_dir"], base_name)  # Create a subdirectory for this video
    
    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Update paths in the config
    config["output_video_path"] = os.path.join(output_dir, f"{base_name}_out_video.mp4")
    config["output_csv_file_path"] = os.path.join(output_dir, f"{base_name}_track_sequences.csv")
    config["output_sequences_dir"] = os.path.join(output_dir, "sequences")
    
    model = YOLO(config["yolo_model"])
    tracker = Tracker(distance_function="euclidean", distance_threshold=100)

    motion_estimator = MotionEstimator(
                max_points=900,
                min_distance=14,
                transformations_getter=HomographyTransformationGetter(),
                draw_flow=False,
            )
    
    tracked_objects = []
    
    print("Generating players detection and tracking data...")

    with video_capture(config["video_path"]) as cap, \
         video_writer(cap, config["output_video_path"]) as out_writer:

        progress_bar = create_progress_bar(cap, desc="Processing Video")
        
        frame_number = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # annotated_frame, tracked_objects = process_frame(frame, 
            #                                                  model, 
            #                                                  tracker, 
            #                                                  motion_estimator, 
            #                                                  tracked_objects)
            
            
            # For some videos it is better to not use a motion estimator
            annotated_frame, tracked_objects = process_frame(frame, 
                                                              model, 
                                                              tracker, 
                                                              None, 
                                                              tracked_objects)
            

            update_tracks_csv(tracked_objects, config["output_csv_file_path"], frame_number)
            
            frame_number += 1
            
            out_writer.write(annotated_frame)
            progress_bar.update(1)

        progress_bar.close()
        
    print("Saving individual players detections for identification...")
    
    sequences = find_continuous_sequences(config["output_csv_file_path"])
    updated_sequences = update_sequences_to_uniform_size(sequences)
    save_sequences_as_videos(updated_sequences, config["video_path"], config["output_sequences_dir"])

if __name__ == "__main__":
    main(CONFIG)