
import cv2
import numpy as np

from utils import homography_transformation_process, draw_visualization_output

def main(config):
    # Calibration and keypoints processing
    guess_fx = 2000  # Default focal length assumption
    guess_rot = np.array([[0.25, 0, 0]])
    guess_trans = (0, 0, 80)
    
    # Homography transformation
    template_img_bgr = cv2.imread(config['input_layout_image'], cv2.IMREAD_COLOR)
    template_img_rgb = cv2.cvtColor(template_img_bgr, cv2.COLOR_BGR2RGB)
    
    # Load numpy keypoints file
    key_points_layout = np.load(config['input_layout_array'])
    
    # Load the original image
    image_bgr = cv2.imread(config['input_image'], cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    
    try:
        # Compute homography process
        H, key_points_filtered, guess_fx, guess_rot, guess_trans = homography_transformation_process(image_rgb, 
                                                                                                     template_img_rgb, 
                                                                                                     key_points_layout, 
                                                                                                     guess_fx, 
                                                                                                     guess_rot, 
                                                                                                     guess_trans)
        if H is None:
            print("Couldn't compute homography")
        else:
            # Compute output and visualize
            im_out = cv2.warpPerspective(image_rgb, H, (template_img_rgb.shape[1], template_img_rgb.shape[0]))
            draw_visualization_output(image_rgb, im_out, H, key_points_filtered)
    except Exception as error:
        # handle the exception
        print(f"An exception occurred: {error}")
    
if __name__ == "__main__":
    for i in range(1,11):
        config = {'input_image': f'../../../../Datasets/soccer_field_homography/test_{i}.png',
                  'input_layout_image': '../../../../Datasets/soccer field layout/soccer_field_layout.png',
                  'input_layout_array': '../../../../Datasets/soccer field layout/soccer_field_layout_points.npy',
                  'output_dir': '../outputs'}
        
        main(config)
    
    