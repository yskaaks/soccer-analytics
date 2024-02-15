

from utils import (create_progress_bar, 
                   update_tracking_csv, 
                   prepare_environment)
from opencv_process import (draw_detected_objects, 
                            video_capture,
                            video_writer,
                            process_and_save_crops)
from yolov8_process import compute_detected_objects
from ultralytics import YOLO
import easyocr
        
def process_frame(frame, model):
    results = model.track(frame, classes=[1, 3], persist=True, verbose=False)
    detected_objects = compute_detected_objects(results, frame)

    return draw_detected_objects(frame, detected_objects, circle_color=(255, 0, 0)), detected_objects

def process_video(config, model, ocr_reader):
    """Process the video, detect objects, save crops, and update tracking info."""
    with video_capture(config["video_path"]) as cap, video_writer(cap, config["output_video_path"]) as out_writer:
        progress_bar = create_progress_bar(cap, desc="Processing Video")
        frame_number = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            annotated_frame, detected_objects = process_frame(frame, model)
            crop_filenames, detected_objects = process_and_save_crops(frame, 
                                                                      detected_objects, 
                                                                      config["output_crops_dir"], 
                                                                      frame_number, 
                                                                      ocr_reader)
            
            update_tracking_csv(config["output_csv_file_path"], frame_number, detected_objects, crop_filenames)
            
            frame_number += 1
            out_writer.write(annotated_frame)
            progress_bar.update(1)
        progress_bar.close()


def main(config):
    prepare_environment(config)
    model = YOLO(config["yolo_model"])
    ocr_reader = easyocr.Reader(['en'], gpu=True)
    print("Generating players detection and tracking data...")
    process_video(config, model, ocr_reader)

if __name__ == "__main__":
    CONFIG = {
        "yolo_model": "../../../Models/pretrained-yolov8-soccer.pt",
        "video_path": "../../../Datasets/soccer_field_homography/video_test_1.mp4",
        "output_base_dir": "../outputs"
    }
    main(CONFIG)
