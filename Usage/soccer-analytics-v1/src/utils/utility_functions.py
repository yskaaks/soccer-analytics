
from tqdm import tqdm
import cv2
import pandas as pd
import os
import shutil
import argparse
from typing import Dict, Any

def create_progress_bar(cap, desc="Processing Video"):
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=total_frames, desc=desc)
    
    return progress_bar

def update_tracking_csv(csv_path, frame_number, detected_objects, crop_filenames):
    """
    Update the tracking CSV file with detected object information and corresponding crop filenames.
    
    Args:
        csv_path: Path to the CSV file.
        frame_number: Current frame number.
        detected_objects: List of detected objects with their bounding box information.
        crop_filenames: List of filenames for the crops of detected objects.
    """
    detections = []
    for det_object, crop_filename in zip(detected_objects, crop_filenames):
        det_id = det_object.id
        x1, y1, x2, y2 = det_object.bbox
        detections.append({
            'frame': frame_number,
            'id': det_id,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'crop_filename': crop_filename  # Include the crop filename
        })
    
    new_df = pd.DataFrame(detections)
    
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        updated_df.to_csv(csv_path, index=False)
    else:
        new_df.to_csv(csv_path, index=False)
        
def prepare_environment(config):
    _reset_directory(config["output_crops_dir"])  # Ensure crops directory is fresh
    _prepare_csv_file(config["output_csv_file_path"])  # Ensure CSV is fresh

def _reset_directory(directory_path):
    """Reset a directory by deleting it if it exists and then recreating it."""
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)

def _prepare_csv_file(csv_path):
    """Prepare the CSV file by deleting it if it exists to start fresh."""
    if os.path.exists(csv_path):
        os.remove(csv_path)
        
def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    Returns an argparse.Namespace object containing the arguments and their values.
    """
    parser = argparse.ArgumentParser(description="Process a video for object detection and tracking.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("--model_path", type=str, default="../../../Models/pretrained-yolov8-soccer.pt", help="Path to the YOLO model file.")
    parser.add_argument("--base_output_dir", type=str, default="../outputs", help="Base directory for output files.")
    
    return parser.parse_args()

def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Loads the configuration based on command-line arguments.
    Args:
    - args: Parsed command-line arguments.
    Returns a configuration dictionary.
    """
    video_filename = os.path.splitext(os.path.basename(args.video_path))[0]
    output_dir = os.path.join(args.base_output_dir, video_filename)
    
    # Constructing dynamic paths based on the input video filename
    config = {
        "yolo_model": args.model_path,
        "video_path": args.video_path,
        "output_video_path": os.path.join(output_dir, f"{video_filename}_processed.mp4"),
        "output_crops_dir": os.path.join(output_dir, "crops"),
        "output_csv_file_path": os.path.join(output_dir, "tracking.csv"),
    }
    
    # Ensure the output directories exist
    os.makedirs(config["output_crops_dir"], exist_ok=True)
    
    return config

if __name__ == "__main__":
    pass