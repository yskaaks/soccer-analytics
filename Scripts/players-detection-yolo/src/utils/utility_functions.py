
from tqdm import tqdm
import cv2
import pandas as pd
import os
import shutil

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
    """Prepare the environment by resetting directories and configuring paths."""
    base_name = os.path.splitext(os.path.basename(config["video_path"]))[0]
    output_dir = os.path.join(config["output_base_dir"], base_name)
    _reset_directory(output_dir)  # Resets the entire output directory for a fresh start
    
    config["output_video_path"] = os.path.join(output_dir, f"{base_name}_out_video.mp4")
    config["output_csv_file_path"] = os.path.join(output_dir, f"{base_name}_track_sequences.csv")
    config["output_crops_dir"] = os.path.join(output_dir, "crops")
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

if __name__ == "__main__":
    pass