"""
This file defines function to be able to spot field lines in order to detect key points
that will help to localize our camera correctly

Try it with

```
python pitch_tracker.py ../images/
```

"""
import cv2
import numpy as np

from .common import intersect
from .key_points import KeyPoints
from .key_lines import KeyLines

def find_extrinsic_intrinsic_matrices(img, guess_fx, guess_rot, guess_trans, key_points):
    """
    Given rough estimate of the focal length and of the camera pose, use PnP algorithm
    to optimally fit key_points (2D) with corresponding points on the pitch (3D)

    This returns the optimal focal length fx and the camera pose.
    """

    height, width = img.shape[0], img.shape[1]

    # Form the problem by associating pixels (2D) with points_world (3D)
    pixels, points_world = key_points.make_2d_3d_association_list()

    # Build camera projection matrix
    fx = key_points.compute_focal_length(guess_fx)

    # Camera projection matrix
    K = np.array([[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]])

    if pixels.shape[0] <= 3:
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
        return None, None, guess_rot, guess_trans

    # in the reference world
    to_device_from_world_rot = cv2.Rodrigues(rotation_vector)[0]

    # to_world_from_device
    camera_position_in_world = -np.matrix(to_device_from_world_rot).T * np.matrix(
        translation_vector
    )
    
    if fx is None:
        return None, None, guess_rot, guess_trans

    dist_to_center = np.linalg.norm(camera_position_in_world)
    if dist_to_center < 40.0 or dist_to_center > 100.0:
        return None, K, guess_rot, guess_trans

    # Build camera pose
    to_device_from_world = np.identity(4)
    to_device_from_world[0:3, 0:3] = to_device_from_world_rot
    to_device_from_world[0:3, 3] = translation_vector.reshape((3,))

    return to_device_from_world, K, rotation_vector, translation_vector


def calibrate_from_image(img, guess_fx, guess_rot, guess_trans, verbose=False):
    """
    After selecting visible key_points, perform PnP algorithm a first time.
    Then, extend key points set by adding not visible corners of the soccer pitch,
    to enforce line fitting.
    Finally redo a PnP pass.
    """

    key_points, key_lines = find_key_points(img)
    
    assert not np.isnan(guess_rot[0, 0])

    to_device_from_world, K, guess_rot, guess_trans = find_extrinsic_intrinsic_matrices(img, 
                                                                                        guess_fx, 
                                                                                        guess_rot, 
                                                                                        guess_trans, 
                                                                                        key_points)
            
    if to_device_from_world is None:
        return K, to_device_from_world, guess_rot, guess_trans, img

    key_points.extend_key_points_set(K, to_device_from_world, key_lines)

    to_device_from_world, K, found_rot, found_trans = find_extrinsic_intrinsic_matrices(img, 
                                                                                        K[0, 0], 
                                                                                        guess_rot, 
                                                                                        guess_trans, 
                                                                                        key_points)

    return K, to_device_from_world, found_rot, found_trans, img


def _find_back_front_lines(img):
    """
    Find back and front lines of the soccer pitch
    (where coach and linesmen are located)
    """

    height = img.shape[0]
    width = img.shape[1]

    # Use a standard canny before performing Hough lines detection
    dst = cv2.Canny(img, 50, 200, None, 3)

    lines = cv2.HoughLines(
        dst,
        1,
        np.pi / 180 / 4,
        500,
        None,
        min_theta=80 / 180 * np.pi,
        max_theta=100 / 180 * np.pi,
    )

    assert lines is not None

    # Loop over the lines to detect only one at the top and at the bottom of the image
    back_line_y = 0
    back_line = None
    front_line_y = height
    front_line = None
    for line in lines:
        rho = line[0][0]
        theta = line[0][1]
        # a = np.cos(theta)
        # b = np.sin(theta)
        y_middle = (rho - width / 2 * np.cos(theta)) / np.sin(theta)
        if back_line_y < y_middle < height / 2:
            back_line_y = y_middle
            back_line = line[0]
        if height / 2 < y_middle < front_line_y:
            front_line_y = y_middle
            front_line = line[0]

    # back and front line are here a tuple (rho, theta) in polar coordinates

    return back_line, front_line


def _remove_out_of_field(img, back_line, front_line):
    """
    Remove every pixels outside of the back and front lines
    i.e. replace them by black pixels
    Return the modified image
    """

    if back_line is None and front_line is None:
        return img

    img_copy = img.copy()

    width = img_copy.shape[1]

    if back_line is None:
        rho_front = front_line[0]
        theta_front = front_line[1]
        for j in range(width):
            y_front = int((rho_front - j * np.cos(theta_front)) / np.sin(theta_front))
            img_copy[y_front:, j] = [0, 0, 0]
        return img_copy

    if front_line is None:
        rho_back = back_line[0]
        theta_back = back_line[1]
        for j in range(width):
            y_back = int((rho_back - j * np.cos(theta_back)) / np.sin(theta_back))
            img_copy[0:y_back, j] = [0, 0, 0]
        return img_copy

    rho_back = back_line[0]
    theta_back = back_line[1]
    rho_front = front_line[0]
    theta_front = front_line[1]
    for j in range(width):
        y_back = int((rho_back - j * np.cos(theta_back)) / np.sin(theta_back))
        y_front = int((rho_front - j * np.cos(theta_front)) / np.sin(theta_front))
        img_copy[0:y_back, j] = [0, 0, 0]
        img_copy[y_front:, j] = [0, 0, 0]
        
    return img_copy


def _find_main_line(img):
    """
    Find the main line (~vertical line on the image)
    Return polar coordinates or None if no main line was found
    """

    # Use a canny/hough line detection to detect a vertical line
    dst = cv2.Canny(img, 50, 200, None, 3)
    lines = cv2.HoughLines(
        dst,
        1,
        np.pi / 180 / 2,
        200,
        None,
        min_theta=0,
        max_theta=40 / 180 * np.pi,
    )

    lines_other = cv2.HoughLines(
        dst,
        1,
        np.pi / 180 / 2,
        250,
        None,
        min_theta=130 / 180 * np.pi,
        max_theta=np.pi,
    )

    if lines is not None:
        return lines[0][0]
    if lines_other is not None:
        return lines_other[0][0]
    return None


def _find_central_circle(
    img, back_middle_point, front_middle_point, main_line, debug=False
):
    """
    Find central circle and return 4 'corners' of this circle.
    Use floodfilling in a Canny image to fill the circle.
    """

    if back_middle_point is None or front_middle_point is None:
        return None, None, None, None

    dst = cv2.Canny(img, 20, 100, None, 3)

    width = img.shape[1]
    if debug:
        cv2.imshow("Canny", dst)
        cv2.waitKey(0)

    # Copy the thresholded image.
    im_floodfill = dst.copy()

    # Dilate to have bolder contours
    im_floodfill = cv2.dilate(im_floodfill, kernel=np.ones((7, 7)))
    if debug:
        cv2.imshow("Dilatation", im_floodfill)
        cv2.waitKey(0)

    # Mask used to flood filling.
    h, w = dst.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from several points where we suspect the circle to be
    back_middle_point = np.array(back_middle_point)
    front_middle_point = np.array(front_middle_point)
    center_approx = 0.3 * front_middle_point + 0.7 * back_middle_point

    for seed in [-150, -100, -50, 50, 100, 150]:
        root = (int(center_approx[0]) + seed, int(center_approx[1]))
        if 0 <= root[0] < width:
            if im_floodfill[root[1], root[0]] != 0:
                continue
            cv2.floodFill(im_floodfill, mask, root, 128)

    if debug:
        cv2.imshow("Floodfill", im_floodfill)
        cv2.waitKey(0)

    final_mask = cv2.inRange(im_floodfill, 127, 129)
    final_mask = cv2.dilate(final_mask, kernel=np.ones((15, 15)))
    final_mask = cv2.erode(final_mask, kernel=np.ones((10, 10)))

    # Find contours around the filled area
    cnts = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv2.contourArea)
    left_circle = tuple(c[c[:, :, 0].argmin()][0])
    right_circle = tuple(c[c[:, :, 0].argmax()][0])
    y_top = tuple(c[c[:, :, 1].argmin()][0])[1]
    y_bottom = tuple(c[c[:, :, 1].argmax()][0])[1]

    behind_circle = [
        int((main_line[0] - y_top * np.sin(main_line[1])) / np.cos(main_line[1])),
        y_top,
    ]
    front_circle = [
        int((main_line[0] - y_bottom * np.sin(main_line[1])) / np.cos(main_line[1])),
        y_bottom,
    ]

    # It can happen that floodfilling was badely executed (wrong seed, little hole in the circle...)
    # so we must discard crazy situations
    if left_circle[0] == 0:
        left_circle = None
    elif left_circle[0] >= behind_circle[0] - 10:
        left_circle = None
    elif left_circle[0] >= front_circle[0] - 10:
        left_circle = None

    if right_circle[0] == img.shape[1] - 1:
        right_circle = None
    elif right_circle[0] <= behind_circle[0] + 10:
        right_circle = None
    elif right_circle[0] <= front_circle[0] + 10:
        right_circle = None

    if left_circle is not None:
        if (front_circle[0] - left_circle[0]) < 2 * (
            front_circle[1] - behind_circle[1]
        ):
            left_circle = None
    if right_circle is not None:
        if (right_circle[0] - front_circle[0]) < 2 * (
            front_circle[1] - behind_circle[1]
        ):
            right_circle = None

    if debug:
        cv2.imshow("FloodFill", img)
        cv2.waitKey(0)

    return left_circle, right_circle, behind_circle, front_circle


def _find_goal_line(img, back_line, back_middle_point, left_line=False):
    """Find goal line based on the back_line position and the center
    left_line specifies if we are looking for the left or the right goal line
    """

    # Use parallel to back line (slightly below) to fit a line only in the area
    # between this line and the back line
    parallel_to_back_line = back_line.copy()
    parallel_to_back_line[0] = back_line[0] + 30 * back_line[1]

    height = img.shape[0]
    width = img.shape[1]

    img_copy = _remove_out_of_field(img, back_line, parallel_to_back_line)

    if back_middle_point is not None:
        if left_line:
            for i in range(height):
                img_copy[i, back_middle_point[0] :] = [0, 0, 0]
        else:
            for i in range(height):
                img_copy[i, : int(back_middle_point[0])] = [0, 0, 0]

    dst = cv2.Canny(img_copy, 50, 200, None, 3)

    min_theta = 0 if left_line else 100 / 180 * np.pi
    max_theta = 80 / 180 * np.pi if left_line else np.pi
    lines = cv2.HoughLines(
        dst,
        1,
        np.pi / 180,
        60,
        None,
        min_theta=min_theta,
        max_theta=max_theta,
    )
    if lines is None:
        return None

    max_y_out = 0

    if left_line:
        for line in lines:
            rho = line[0][0]
            theta = line[0][1]
            y_out = rho / np.sin(theta)
            if y_out > max_y_out:
                max_y_out = y_out
                goal_line = line[0]
    else:
        for line in lines:
            rho = line[0][0]
            theta = line[0][1]
            y_out = (rho - width * np.cos(theta)) / np.sin(theta)
            if y_out > max_y_out:
                max_y_out = y_out
                goal_line = line[0]

    intersection_with_back_line_first = intersect(back_line, goal_line)
    dst = cv2.Canny(img, 50, 200, None, 3)
    # Second pass with larger hough detection
    lines = cv2.HoughLines(
        dst,
        1,
        np.pi / 180,
        200,
        None,
        min_theta=min_theta,
        max_theta=max_theta,
    )

    first_goal_line = goal_line.copy()

    if lines is None:
        return first_goal_line

    min_dist = 1000
    for line in lines:
        theta = line[0][1]
        if abs(theta - first_goal_line[1]) > 0.03:
            continue

        intersection_with_back_line = intersect(back_line, line[0])
        dist = np.linalg.norm(
            np.array(intersection_with_back_line)
            - np.array(intersection_with_back_line_first)
        )
        if dist < min_dist:
            goal_line = line[0]
            min_dist = dist

    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    return goal_line


def find_key_points(img):
    """
    Find key points in image
    """
    width = img.shape[1]

    key_lines = KeyLines()

    key_lines.back_line, key_lines.front_line = _find_back_front_lines(img)

    img_wo_out_of_field = _remove_out_of_field(
        img, key_lines.back_line, key_lines.front_line
    )

    key_lines.main_line = _find_main_line(img_wo_out_of_field)

    # key points can be found at the intersection
    key_points = KeyPoints()

    key_points.back_middle_line = intersect(key_lines.main_line, key_lines.back_line)
    key_points.front_middle_line = intersect(key_lines.main_line, key_lines.front_line)

    # Find central circle
    (
        key_points.left_circle,
        key_points.right_circle,
        key_points.behind_circle,
        key_points.front_circle,
    ) = _find_central_circle(
        img,
        key_points.back_middle_line,
        key_points.front_middle_line,
        key_lines.main_line,
    )

    # Last step: detect goal line if there is one showing up
    if (
        key_points.back_middle_line is not None
        and key_points.back_middle_line[0] < width * 2 / 5
    ):
        key_lines.right_goal_line = _find_goal_line(
            img, key_lines.back_line, key_points.back_middle_line, False
        )

    if (
        key_points.back_middle_line is not None
        and key_points.back_middle_line[0] > width * 3 / 5
    ):
        key_lines.left_goal_line = _find_goal_line(
            img, key_lines.back_line, key_points.back_middle_line, True
        )

    if key_lines.right_goal_line is not None:
        key_points.corner_back_right = intersect(
            key_lines.right_goal_line, key_lines.back_line
        )
    if key_lines.left_goal_line is not None:
        key_points.corner_back_left = intersect(
            key_lines.left_goal_line, key_lines.back_line
        )

    return key_points, key_lines


if __name__ == "__main__":
    pass
