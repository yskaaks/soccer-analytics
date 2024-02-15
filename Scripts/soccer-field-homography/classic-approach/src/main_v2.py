
import cv2
import matplotlib.pyplot as plt
import numpy as np

from pitch_tracker.main import find_key_points

from camera_pose_estimation.projector import (project_to_screen)

from pitch_tracker.key_points import (corner_front_right_world, 
                                      corner_front_left_world, 
                                      corner_back_right_world, 
                                      corner_back_left_world)

from pitch_tracker.common import intersect

from camera_pose_estimation.projector import (draw_pitch_lines)

def find_extrinsic_intrinsic_matrices(img, guess_fx, guess_rot, guess_trans, key_points):
    """
    Given rough estimate of the focal length and of the camera pose, use PnP algorithm
    to optimally fit key_points (2D) with corresponding points on the pitch (3D)

    This returns the optimal focal length fx and the camera pose.
    """

    height, width = img.shape[0], img.shape[1]

    # Form the problem by associating pixels (2D) with points_world (3D)
    pixels, points_world = key_points.make_2d_3d_association_list()

    # PnP algo needs at least 4 points to work
    print(f"Solving PnP with {len(pixels)} points")

    # Build camera projection matrix
    fx = key_points.compute_focal_length(guess_fx)

    # Camera projection matrix
    K = np.array([[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]])

    if pixels.shape[0] <= 3:
        print("Too few points to solve!")
        return None, K, guess_rot, guess_trans

    # Perspective-n-Point algorithm, returning rotation and translation vector
    (ret, rotation_vector, translation_vector) = cv2.solvePnP(
        points_world,
        pixels,
        K,
        distCoeffs=None,
        rvec=guess_rot,
        tvec=guess_trans,
        useExtrinsicGuess=True,
    )

    assert ret

    if np.isnan(rotation_vector[0, 0]):
        print("PnP could not be solved correctly --> Skip")
        return None, None, guess_rot, guess_trans

    # in the reference world
    to_device_from_world_rot = cv2.Rodrigues(rotation_vector)[0]

    # to_world_from_device
    camera_position_in_world = -np.matrix(to_device_from_world_rot).T * np.matrix(
        translation_vector
    )

    print(
        f"Camera is located at {-camera_position_in_world[1,0]:.1f}m high and "
        f"at {-camera_position_in_world[2,0]:.1f}m depth"
    )
    if fx is None:
        print(f"PnP outputed crazy value for focal length: {fx} --> Skip")
        return None, None, guess_rot, guess_trans

    dist_to_center = np.linalg.norm(camera_position_in_world)
    print(f"Final fx = {fx:.1f}. Distance to origin = {dist_to_center:.1f}m")
    if dist_to_center < 40.0 or dist_to_center > 100.0:
        print(
            f"PnP outputed crazy value for distance to center = {dist_to_center:.1f}m --> Skip"
        )
        return None, K, guess_rot, guess_trans

    # Build camera pose
    to_device_from_world = np.identity(4)
    to_device_from_world[0:3, 0:3] = to_device_from_world_rot
    to_device_from_world[0:3, 3] = translation_vector.reshape((3,))

    return to_device_from_world, K, rotation_vector, translation_vector

def find_closer_point_on_line(point, line):
    """Find closer point on a line to the given point"""
    rho = line[0]
    theta = line[1]
    point = np.array(point)

    pt_line_origin = np.array([0, rho / np.sin(theta)])
    a = point - pt_line_origin
    u = np.array([np.sin(theta), -np.cos(theta)])

    _lambda = np.dot(a, u)

    projected_point = pt_line_origin + _lambda * u

    projected_point = [int(projected_point[0]), int(projected_point[1])]

    return projected_point

def extend_key_points_set(key_points, K, to_device_from_world, key_lines):
    """
    As the PnP is not so performant with only a few points, we try to get closer
    to the Perspective-n-Line algo by projecting corners even if they are not visible

    We modifiy the key_points set.
    """

    if key_points.corner_back_right is None and key_points.corner_back_left is None:
        pt = project_to_screen(K, to_device_from_world, corner_front_right_world)
        key_points.corner_front_right = find_closer_point_on_line(
            pt, key_lines.front_line
        )
        pt = project_to_screen(K, to_device_from_world, corner_front_left_world)
        key_points.corner_front_left = find_closer_point_on_line(
            pt, key_lines.front_line
        )
        pt = project_to_screen(K, to_device_from_world, corner_back_right_world)
        key_points.corner_back_right = find_closer_point_on_line(
            pt, key_lines.back_line
        )
        pt = project_to_screen(K, to_device_from_world, corner_back_left_world)
        key_points.corner_back_left = find_closer_point_on_line(pt, key_lines.back_line)

    if (
        key_points.corner_back_right is not None
        and key_lines.right_goal_line is not None
    ):
        key_points.corner_front_right = intersect(
            key_lines.right_goal_line, key_lines.front_line
        )
    if key_points.corner_back_left is not None and key_lines.left_goal_line is not None:
        key_points.corner_front_left = intersect(
            key_lines.left_goal_line, key_lines.front_line
        )
        
def print_verbose_arrays(verbose_dir):
    print("\n---------Verbose information:")
    for key in verbose_dir.keys():
        if verbose_dir[key] is not None:
            print(f"Shape of {key}: {verbose_dir[key].shape}")
        else:
            print(f"Shape of {key}: None")

def calibrate_from_image(img, guess_fx, guess_rot, guess_trans, verbose=False):
    """
    After selecting visible key_points, perform PnP algorithm a first time.
    Then, extend key points set by adding not visible corners of the soccer pitch,
    to enforce line fitting.
    Finally redo a PnP pass.
    """

    key_points, key_lines = find_key_points(img)
    
    print("\n---------Obtained Keypoints:")
    print(key_points)

    assert not np.isnan(guess_rot[0, 0])

    to_device_from_world, K, guess_rot, guess_trans = find_extrinsic_intrinsic_matrices(img, 
                                                                                        guess_fx, 
                                                                                        guess_rot, 
                                                                                        guess_trans, 
                                                                                        key_points)

    verbose_dir = {'to_device_from_world': to_device_from_world, 
                   'K': K, 
                   'guess_rot': guess_rot, 
                   'guess_trans': guess_trans}
    
    print_verbose_arrays(verbose_dir)
            
    if to_device_from_world is None:
        return K, to_device_from_world, guess_rot, guess_trans, img

    extend_key_points_set(key_points, K, to_device_from_world, key_lines)
    
    print("\n---------Extended Keypoints:")
    print(key_points)

    to_device_from_world, K, found_rot, found_trans = find_extrinsic_intrinsic_matrices(img, 
                                                                                        K[0, 0], 
                                                                                        guess_rot, 
                                                                                        guess_trans, 
                                                                                        key_points)
    
    verbose_dir = {'Updated to_device_from_world': to_device_from_world, 
                   'Updated K': K, 
                   'New found_rot': found_rot, 
                   'New found_trans': found_trans}

    print_verbose_arrays(verbose_dir)

    return K, to_device_from_world, found_rot, found_trans, img

def display_yaw_and_focal_length(img, yaw, fx):
    """Display infos on image (yaw angle + fx)"""
    img = cv2.putText(
        img,
        f"Yaw: {yaw:.0f} deg, Focal: {fx:.0f}",
        (1280, 120),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        color=(0, 255, 0),
        thickness=2,
    )

    return img

def compute_homography_matrix(pts_src, pts_dst):
    mask = ~np.any(np.isnan(pts_src), axis=1)

    pts_src_filtered = pts_src[mask]
    pts_dst_filtered = pts_dst[mask]

    h, status = cv2.findHomography(pts_src_filtered, pts_dst_filtered)
    
    return h, pts_src_filtered

def apply_homography_to_point(H, point):
    # Convert point to homogeneous coordinates
    point_homogeneous = np.array([point[0], point[1], 1]).reshape(-1, 1)
    
    # Apply the homography matrix
    point_transformed_homogeneous = np.dot(H, point_homogeneous)
    
    # Convert back to Cartesian coordinates
    point_transformed = point_transformed_homogeneous[:2] / point_transformed_homogeneous[2]
    
    return point_transformed.flatten()

def apply_homography_to_array(H, points):
    # Convert points to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack([points, ones])
    
    # Apply the homography matrix
    points_transformed_homogeneous = np.dot(H, points_homogeneous.T).T
    
    # Convert back to Cartesian coordinates
    points_transformed = points_transformed_homogeneous[:, :2] / points_transformed_homogeneous[:, 2][:, np.newaxis]
    
    return points_transformed

def draw_points_on_image(image, points):
    for (x, y) in points:
        cv2.circle(image, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1)
        
    return image

def main(config):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Adjust subplot layout to 1 row, 3 columns

    # Load the original image
    image_bgr = cv2.imread(config['input_image'], cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Display the original image
    axs[0].imshow(image_rgb)
    axs[0].set_title("Input Image", fontweight='bold')
    axs[0].axis('off')
    
    # Calibration and keypoints processing
    guess_fx = 2000  # Default focal length assumption
    guess_rot = np.array([[0.25, 0, 0]])
    guess_trans = (0, 0, 80)
    K, to_device_from_world, rot, trans, _ = calibrate_from_image(image_rgb, guess_fx, guess_rot, guess_trans)
    if to_device_from_world is not None:
        image_rgb_with_pitch_lines = draw_pitch_lines(K, to_device_from_world, image_rgb.copy())
        image_rgb_with_pitch_lines = display_yaw_and_focal_length(image_rgb_with_pitch_lines, guess_rot[0, 1] * 180 / np.pi, K[0, 0])
    else:
        image_rgb_with_pitch_lines = image_rgb
    
    # Homography transformation
    template_img_bgr = cv2.imread(config['input_layout_image'], cv2.IMREAD_COLOR)
    template_img_rgb = cv2.cvtColor(template_img_bgr, cv2.COLOR_BGR2RGB)
    key_points_layout = np.load(config['input_layout_array'])
    key_points, _ = find_key_points(image_rgb)
    H, key_points_filtered = compute_homography_matrix(key_points.compute_points_array(), key_points_layout)
    im_out = cv2.warpPerspective(image_rgb, H, (template_img_rgb.shape[1], template_img_rgb.shape[0]))
    
    
    image_rgb_with_pitch_lines = draw_points_on_image(image_rgb_with_pitch_lines, key_points_filtered)
    # Display image with keypoints and pitch lines
    axs[1].imshow(image_rgb_with_pitch_lines)
    axs[1].set_title("Keypoints & Pitch Lines", fontweight='bold')
    axs[1].axis('off')
    
    # Transform filtered points with homography matrix
    key_points_filtered_transformed = apply_homography_to_array(H, key_points_filtered)
    im_out = draw_points_on_image(im_out, key_points_filtered_transformed)
    # Display transformed image (homography output)
    axs[2].imshow(im_out)
    axs[2].set_title("Transformed Image", fontweight='bold')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

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
    
    
    verbose_dir = {'guess_rot': guess_rot, 
                   'guess_trans': guess_trans,
                   'guess_fx': guess_fx}
    
    print_verbose_arrays(verbose_dir)
    
if __name__ == "__main__":
    for i in range(1,10):
        config = {'input_image': f'../../../../Datasets/soccer_field_homography/test_{i}.png',
                  'input_layout_image': '../../../../Datasets/soccer field layout/soccer_field_layout.png',
                  'input_layout_array': '../../../../Datasets/soccer field layout/soccer_field_layout_points.npy',
                  'output_dir': '../outputs'}
        
        main(config)
    
    