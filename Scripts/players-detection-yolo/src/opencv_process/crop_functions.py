

import os
import cv2
import re

def process_and_save_crops(frame, detected_objects, output_crops_dir, frame_number, ocr_reader):
    """Process and save crops of the current frame based on detected objects' bounding boxes."""
    crop_filenames = []
    for i, det_object in enumerate(detected_objects):
        crop_filename, crop_path, det_object = _save_crop(frame, 
                                                          det_object, 
                                                          output_crops_dir, 
                                                          frame_number, 
                                                          ocr_reader)
        detected_objects[i] = det_object
        
        
        crop_filenames.append(crop_filename)
    return crop_filenames, detected_objects

def _save_crop(frame, det_object, output_crops_dir, frame_number, ocr_reader):
    """Save a single crop of a detected object."""
    x1, y1, x2, y2 = [int(coord) for coord in det_object.bbox]
    crop = frame[y1:y2, x1:x2]
    
    # Experimental id assigning using EasyOCR
    det_object.id = float('nan')
    ocr_detections = ocr_reader.readtext(crop, detail = 1)
    
    if len(ocr_detections) > 0:
        ocr_bbox, text, score = ocr_detections[0]
        # Sanitize text for filename
        sanitized_text = re.sub(r'[^a-zA-Z0-9]+', '_', text)[:10]
        
        if score > 0.8:
            if sanitized_text.isdigit():
                det_object.id = int(sanitized_text)
        
        # Crop the image based on the bounding box and save it in the confident_crops directory
        top_left = tuple(map(int, ocr_bbox[0]))
        bottom_right = tuple(map(int, ocr_bbox[2]))
        ocr_crop = crop[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        ocr_crop_filename = f"OCR_frame{frame_number}_id{det_object.id}_{x1}_{y1}_{x2}_{y2}_{sanitized_text}_{score:.2f}.jpg"
        
        cv2.imwrite(os.path.join(output_crops_dir, ocr_crop_filename), ocr_crop)
        
    det_id = det_object.id
    
    crop_filename = f"frame{frame_number}_id{det_id}_{x1}_{y1}_{x2}_{y2}.jpg"
    crop_path = os.path.join(output_crops_dir, crop_filename)
    cv2.imwrite(crop_path, crop)
    return crop_filename, crop_path, det_object

if __name__ == "__main__":
    pass