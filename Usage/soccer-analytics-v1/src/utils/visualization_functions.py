

import cv2
import matplotlib.pyplot as plt

from .homography_functions import apply_homography_to_array

def draw_points_on_image(image, points):
    for (x, y) in points:
        cv2.circle(image, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1)
        
    return image

def draw_visualization_output(im_in, im_out, H, keypoints):
    # Draw original keypoints
    im_in = draw_points_on_image(im_in, keypoints)
    
    # Transform filtered points with homography matrix
    keypoints_transformed = apply_homography_to_array(H, keypoints)
    im_out = draw_points_on_image(im_out, keypoints_transformed)
    
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))  # Adjust subplot layout to 1 row, 3 columns
    # Display the original image
    axs[0].imshow(im_in)
    axs[0].set_title("Input Image", fontweight='bold')
    axs[0].axis('off')
    
    # Display transformed image (homography output)
    axs[1].imshow(im_out)
    axs[1].set_title("Transformed Image", fontweight='bold')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    pass