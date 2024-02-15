"""
This file aims at estimating optimal camera calibration matrices based only on
the detected keypoints of the pitch.
We assume that the camera has no distortion.
We assume that intrinsic matrix is unknown (no pre-calibration), so we assume that the
optical center is perfectly centered, that the focal length is the same along x and y axis.
We have an initial rough estimate of the camera position, but we let the 6 degrees
of freedom to find the camera pose (relative to the pitch)

You can test it with:

```
python camera_pose_estimation/main.py images/
```

"""
from argparse import ArgumentParser
from pathlib import Path
import cv2
import numpy as np

from camera_pose_estimation.projector import (
    draw_pitch_lines,
    project_to_screen,
)

from pitch_tracker.common import intersect
from pitch_tracker.key_points import (
    corner_front_right_world,
    corner_front_left_world,
    corner_back_right_world,
    corner_back_left_world,
)
from pitch_tracker.main import find_key_points


def find_extrinsic_intrinsic_matrices(
    img, guess_fx, guess_rot, guess_trans, key_points
):
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


def calibrate_from_image(img, guess_fx, guess_rot, guess_trans):
    """
    After selecting visible key_points, perform PnP algorithm a first time.
    Then, extend key points set by adding not visible corners of the soccer pitch,
    to enforce line fitting.
    Finally redo a PnP pass.
    """

    key_points, key_lines = find_key_points(img)

    assert not np.isnan(guess_rot[0, 0])

    to_device_from_world, K, guess_rot, guess_trans = find_extrinsic_intrinsic_matrices(
        img, guess_fx, guess_rot, guess_trans, key_points
    )

    if to_device_from_world is None:
        return K, to_device_from_world, guess_rot, guess_trans, img

    extend_key_points_set(key_points, K, to_device_from_world, key_lines)

    to_device_from_world, K, found_rot, found_trans = find_extrinsic_intrinsic_matrices(
        img, K[0, 0], guess_rot, guess_trans, key_points
    )

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


def display_top_view(K, to_device_from_world, img):
    """
    Display top view of the image, by unprojecting every pixels of the image
    to its correspondance in the world, assuming that altitude is 0
    """

    height = img.shape[0]
    width = img.shape[1]

    # Top view
    width_soccer_field = 105
    height_soccer_field = 68
    unskewd = np.zeros((height, width, 3), np.uint8)
    pixels_width = int(width_soccer_field / height_soccer_field * height)
    offset_x = (width - pixels_width) // 2

    for i in range(height):
        z_world = -corner_front_left_world[2] - i * height_soccer_field / height
        for j in range(1668):

            x_world = corner_front_left_world[0] + j * width_soccer_field / pixels_width
            proj = project_to_screen(
                K, to_device_from_world, np.array([x_world, 0, z_world])
            )
            u = proj[0]
            v = proj[1]
            if 0 <= u < width and 0 <= v < height:
                unskewd[i, j + offset_x] = img[v, u]

    return unskewd


if __name__ == "__main__":
    # Default value of our focal length to start with
    # Dont forget to change this value if you are using different sizes of images
    guess_fx = 2000

    parser = ArgumentParser(
        description="Main script to key points on a soccer field image"
    )
    parser.add_argument("input", type=str, help="Image path or folder")
    parser.add_argument(
        "--out",
        action="store_const",
        const=True,
        default=False,
        help="Output images with pose estimation",
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        raise FileExistsError

    images = (
        sorted(Path(args.input).glob("**/*"))
        if Path(args.input).is_dir()
        else [Path(args.input)]
    )

    guess_rot = np.array([[0.25, 0, 0]])
    guess_trans = (0, 0, 80)

    for filename in images:
        if not filename.exists():
            continue

        print(str(filename))
        img = cv2.imread(str(filename))

        K, to_device_from_world, rot, trans, img = calibrate_from_image(
            img, guess_fx, guess_rot, guess_trans
        )

        # Uncomment to have unskwed image (top view)
        # if to_device_from_world is not None:
        #     unskewd = display_top_view(K, to_device_from_world, img)
        #     out_path = Path("top_views").joinpath(Path(filename).name)
        #     print(f"Writing image to {str(out_path)}")
        #     cv2.imwrite(str(out_path), unskewd)
        
        #     cv2.imshow("unskewd", unskewd)
        #     cv2.waitKey(0)

        key_points, key_lines = find_key_points(img)
        img = key_points.draw(img)

        if to_device_from_world is not None:
            img = draw_pitch_lines(K, to_device_from_world, img)
            img = display_yaw_and_focal_length(
                img, guess_rot[0, 1] * 180 / np.pi, K[0, 0]
            )

        # Modify current value of calibration matrices to get benefit
        # of this computation for next image
        guess_rot = (
            rot if to_device_from_world is not None else np.array([[0.25, 0, 0]])
        )
        guess_trans = trans if to_device_from_world is not None else (0, 0, 80)
        guess_fx = K[0, 0]

        if not args.out:
            cv2.imshow("Test", img)
            k = cv2.waitKey(0)
            if k == 27:
                break
        else:
            out_path = Path("camera_pose_estimation/out").joinpath(Path(filename).name)
            print(f"Writing image to {str(out_path)}")
            cv2.imwrite(str(out_path), img)
