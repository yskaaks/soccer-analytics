

import numpy as np
import cv2
from pitch_tracker import find_key_points, calibrate_from_image

def homography_transformation_process(image_rgb, template_img_rgb, key_points_layout, guess_fx, guess_rot, guess_trans):
    K, to_device_from_world, rot, trans, _ = calibrate_from_image(image_rgb, guess_fx, guess_rot, guess_trans)
    
    key_points, _ = find_key_points(image_rgb)
    
    H, key_points_filtered = _compute_homography_matrix(key_points.compute_points_array(), key_points_layout)
    
    # Update calibration matrices for next image
    guess_rot = rot if to_device_from_world is not None else np.array([[0.25, 0, 0]])
    guess_trans = trans if to_device_from_world is not None else (0, 0, 80)
    guess_fx = K[0, 0]
    
    
    # Modify current value of calibration matrices to get benefit
    # of this computation for next image
    guess_rot = (
        rot if to_device_from_world is not None else np.array([[0.25, 0, 0]])
    )
    guess_trans = trans if to_device_from_world is not None else (0, 0, 80)
    guess_fx = K[0, 0]
    
    return H, key_points_filtered, guess_fx, guess_rot, guess_trans

def apply_homography_to_array(H, points):
    # Convert points to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack([points, ones])
    
    # Apply the homography matrix
    points_transformed_homogeneous = np.dot(H, points_homogeneous.T).T
    
    # Convert back to Cartesian coordinates
    points_transformed = points_transformed_homogeneous[:, :2] / points_transformed_homogeneous[:, 2][:, np.newaxis]
    
    return points_transformed

def _compute_homography_matrix(pts_src, pts_dst):
    mask = ~np.any(np.isnan(pts_src), axis=1)

    pts_src_filtered = pts_src[mask]
    pts_dst_filtered = pts_dst[mask]

    h, status = cv2.findHomography(pts_src_filtered, pts_dst_filtered)
    
    return h, pts_src_filtered

if __name__ == "__main__":
    pass
