
import cv2
import numpy as np
import os
import logging
from typing import Tuple

# Assuming 'homography_transformation_process' function is defined elsewhere
from utils import homography_transformation_process
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HomographyState:
    """
    Class to maintain the state and parameters for homography transformation across video frames.
    """
    def __init__(self, guess_fx: float, guess_rot: np.ndarray, guess_trans: Tuple[float, float, float]):
        self.guess_fx = guess_fx
        self.guess_rot = guess_rot
        self.guess_trans = guess_trans

    def update_state(self, guess_fx: float, guess_rot: np.ndarray, guess_trans: Tuple[float, float, float]):
        """
        Update the state with new values from the latest homography transformation process.
        """
        self.guess_fx = guess_fx
        self.guess_rot = guess_rot
        self.guess_trans = guess_trans

class VideoProcessor:
    """
    Class to handle video processing and homography transformations.
    """
    def __init__(self, config: dict):
        self.config = config
        self.state = HomographyState(guess_fx=2000, guess_rot=np.array([[0.25, 0, 0]]), guess_trans=(0, 0, 80))
        self.template_img_rgb = self.load_template_image(self.config['input_layout_image'])
        self.key_points_layout = np.load(self.config['input_layout_array'])

    @staticmethod
    def load_template_image(path: str) -> np.ndarray:
        """
        Load and return the template image in RGB format.
        """
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

    @staticmethod
    def proportion_of_black_pixels(img: np.ndarray) -> float:
        """
        Calculate and return the proportion of black pixels in the image.
        """
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        black_pixel_count = np.sum(gray_img == 0)
        total_pixels = gray_img.size
        return black_pixel_count / total_pixels

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single video frame and return the transformed frame.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            # Assuming 'homography_transformation_process' is implemented and available
            H, key_points_filtered, guess_fx, guess_rot, guess_trans = homography_transformation_process(
                frame_rgb, self.template_img_rgb, self.key_points_layout, 
                self.state.guess_fx, self.state.guess_rot, self.state.guess_trans)
            
            if H is None:
                logging.warning("Couldn't compute homography")
                return np.zeros_like(self.template_img_rgb)
            
            im_out = cv2.warpPerspective(frame_rgb, H, (self.template_img_rgb.shape[1], self.template_img_rgb.shape[0]))
            im_out_bgr = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)
            
            if self.proportion_of_black_pixels(im_out_bgr) < 0.1:
                return np.zeros_like(self.template_img_rgb)

            self.state.update_state(guess_fx, guess_rot, guess_trans)
            return im_out_bgr
        except Exception as e:
            logging.error(f"An exception occurred: {e}")
            return np.zeros_like(self.template_img_rgb)

    def process_video(self):
        """
        Process the entire video and save the output.
        """
        cap = cv2.VideoCapture(self.config['input_video_path'])
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = self.construct_output_path(self.config['input_video_path'])
        out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (self.template_img_rgb.shape[1], self.template_img_rgb.shape[0]))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with tqdm(total=total_frames, desc="Processing video frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    processed_frame = self.process_frame(frame)
                    out.write(processed_frame)
                    pbar.update(1)
                else:
                    break
        
        cap.release()
        out.release()

    @staticmethod
    def construct_output_path(input_video_path: str) -> str:
        """
        Construct and return the output video path based on the input video path.
        """
        input_video_filename = os.path.basename(input_video_path)
        output_video_filename = os.path.splitext(input_video_filename)[0] + ".mp4"
        return os.path.join('../outputs', output_video_filename)

if __name__ == "__main__":
    config = {
        'input_video_path': '../../../../Datasets/soccer_field_homography/video_test_1.mp4',
        'input_layout_image': '../../../../Datasets/soccer field layout/soccer_field_layout.png',
        'input_layout_array': '../../../../Datasets/soccer field layout/soccer_field_layout_points.npy'
    }
    processor = VideoProcessor(config)
    processor.process_video()

    
    