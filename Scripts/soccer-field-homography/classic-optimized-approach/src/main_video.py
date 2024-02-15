

import cv2
import numpy as np

from utils import homography_transformation_process
from tqdm import tqdm

import os

# Define a class to maintain state across frames
class HomographyState:
    def __init__(self, guess_fx, guess_rot, guess_trans):
        self.guess_fx = guess_fx
        self.guess_rot = guess_rot
        self.guess_trans = guess_trans
        self.previous_blk_count = None
        
def proportion_of_black_pixels(img):
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Count the number of black pixels
    black_pixel_count = np.sum(gray_img == 0)
    
    # Calculate the total number of pixels in the image
    total_pixels = gray_img.size
    
    # Calculate the proportion of black pixels
    black_pixel_proportion = black_pixel_count / total_pixels
    
    return black_pixel_proportion

def process_frame(frame, template_img_rgb, key_points_layout, state):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        # Compute homography process with updated guesses
        H, key_points_filtered, state.guess_fx, state.guess_rot, state.guess_trans = homography_transformation_process(
            frame_rgb, 
            template_img_rgb, 
            key_points_layout, 
            state.guess_fx, 
            state.guess_rot, 
            state.guess_trans)
        if H is None:
            print("Couldn't compute homography")
            # Return a black image with the same shape as template_img_rgb
            return np.zeros_like(template_img_rgb)
        else:
            # Apply transformation and convert back to BGR for video output
            im_out = cv2.warpPerspective(frame_rgb, H, (template_img_rgb.shape[1], template_img_rgb.shape[0]))
            im_out_bgr = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)
            
            
            if proportion_of_black_pixels(im_out_bgr) < 0.1:
                return np.zeros_like(template_img_rgb)

            return im_out_bgr
    except Exception as error:
        print(f"An exception occurred: {error}")
        # Return a black image with the same shape as template_img_rgb in case of an exception
        return np.zeros_like(template_img_rgb)

def main(config):
    # Initialize homography state with default values
    state = HomographyState(guess_fx=2000, guess_rot=np.array([[0.25, 0, 0]]), guess_trans=(0, 0, 80))
    
    # Homography transformation setup
    template_img_bgr = cv2.imread(config['input_layout_image'], cv2.IMREAD_COLOR)
    template_img_rgb = cv2.cvtColor(template_img_bgr, cv2.COLOR_BGR2RGB)
    key_points_layout = np.load(config['input_layout_array'])
    
    # Video processing setup
    cap = cv2.VideoCapture(config['input_video_path'])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Dynamically construct output_video_path based on input_video_path
    input_video_filename = os.path.basename(config['input_video_path'])  # Extract the filename from the input path
    output_video_filename = os.path.splitext(input_video_filename)[0] + ".mp4"  # Construct the output filename
    output_video_path = os.path.join('../outputs', output_video_filename)  # Construct the full output path
    
    # Adjust the output video's resolution to match that of the template_img_rgb
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (template_img_rgb.shape[1], template_img_rgb.shape[0]))
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames in the video
    
    # Process each frame with a tqdm progress bar
    with tqdm(total=total_frames, desc="Processing video frames") as pbar:
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                processed_frame = process_frame(frame, template_img_rgb, key_points_layout, state)
                out.write(processed_frame)
                pbar.update(1)  # Update progress bar by one step per frame
            else:
                break

    cap.release()
    out.release()
    
if __name__ == "__main__":
    config = {
        'input_video_path': '../../../../Datasets/soccer_field_homography/video_test_1.mp4',
        'input_layout_image': '../../../../Datasets/soccer field layout/soccer_field_layout.png',
        'input_layout_array': '../../../../Datasets/soccer field layout/soccer_field_layout_points.npy'
    }
    main(config)

    
    