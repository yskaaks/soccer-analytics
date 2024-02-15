import easyocr
import cv2
import os
import re
from tqdm import tqdm

def process_images_from_folder(input_dir, output_base_dir, reader):
    # Ensure the output base directory exists
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    # Create the output directory named after the input directory
    output_dir = os.path.join(output_base_dir, os.path.basename(input_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a subdirectory for confident crops
    confident_crops_dir = os.path.join(output_dir, "confident_crops")
    if not os.path.exists(confident_crops_dir):
        os.makedirs(confident_crops_dir)

    # Iterate over each image in the input directory
    images = [img for img in os.listdir(input_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for index, image_name in tqdm(enumerate(images), total=len(images)):
        image_path = os.path.join(input_dir, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Image {image_name} not found.")
            continue

        detections = reader.readtext(image, detail = 1)

        # for detection in detections:
        if len(detections) > 0:
            bbox, text, score = detections[0]
            # Sanitize text for filename
            sanitized_text = re.sub(r'[^a-zA-Z0-9]+', '_', text)[:10]
            if score > 0.8:
                # Crop the image based on the bounding box and save it in the confident_crops directory
                top_left = tuple(map(int, bbox[0]))
                bottom_right = tuple(map(int, bbox[2]))
                crop_img = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                crop_filename = f"{index}_{sanitized_text}_{score:.2f}.jpg"
                cv2.imwrite(os.path.join(confident_crops_dir, crop_filename), crop_img)
        else:
            sanitized_text = "None"
            score = 0.0

        # Optionally, save the original image in the output directory with the new naming convention
        new_image_name = f"{index}_{sanitized_text}_{score:.2f}.jpg"  # Original image saved with index as filename
        cv2.imwrite(os.path.join(output_dir, new_image_name), image)
if __name__ == "__main__":
    input_dir = "C:/Users/anlun/OneDrive/Documents/GitHub/soccer-players-analysis/Scripts/players-detection-yolo/outputs/test_soccer_video_4/crops"  # Update this path to your input folder
    output_base_dir = "../outputs"  # Update this path to your desired output base directory
    
    reader = easyocr.Reader(['en'], gpu=True)  # Set `gpu=False` if you're not using GPU
    process_images_from_folder(input_dir, output_base_dir, reader)
