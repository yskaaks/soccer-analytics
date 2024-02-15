"""
This module defines KeyPoints class and its correspondance in the 3D world
See the README.md to have a better understanding of the naming of the 3D points
"""

import numpy as np
import cv2
from .common import draw_point
from .common import intersect

# Points in world with origin au centre, x going right, y to the foreground, z to the top
# Positions are based on official soccer dimensions
right_circle_world = [9.15, 0, 0]
left_circle_world = [-9.15, 0, 0]
behind_circle_world = [0, 0, 9.15]
front_circle_world = [0, 0, -9.15]
front_middle_line_world = [0, 0, -34]
back_middle_line_world = [0, 0, 34]

corner_back_left_world = [-52.5, 0, 34]
corner_front_left_world = [-52.5, 0, -34]
corner_back_right_world = [52.5, 0, 34]
corner_front_right_world = [52.5, 0, -34]


DIST_TO_CENTER = 77.0

def project_to_screen(K, to_device_from_world, point_in_world):
    """
    Project point_in_world to the screen and returns pixel coordinates
    """
    homog = np.ones((4, 1))
    homog[0:3, 0] = point_in_world
    point_in_device = np.dot(to_device_from_world, homog)
    point_in_device_divided_depth = (point_in_device / point_in_device[2, 0])[0:3]
    point_projected = np.dot(K, point_in_device_divided_depth)
    return [int(point_projected[0]), int(point_projected[1])]

def draw_pitch_lines(K, to_device_from_world, img):
    """Draw pitch lines"""
    img = _draw_central_circle(K, to_device_from_world, img)
    img = _draw_middle_line(K, to_device_from_world, img)
    img = _draw_border_lines(K, to_device_from_world, img)
    img = _draw_penalty_areas(K, to_device_from_world, img)

    return img

def _project_and_draw_lines(K, to_device_from_world, points_in_world, img):
    """Project list of points in world and draw polyline"""

    projected_points = []
    for point_in_world in points_in_world:
        projected_points.append(
            project_to_screen(K, to_device_from_world, np.array(point_in_world))
        )

    nb_pts = len(projected_points)
    for i in range(nb_pts):
        img = cv2.line(
            img,
            projected_points[i],
            projected_points[(i + 1) % nb_pts],
            color=(0, 165, 255),
            thickness=3,
        )

    return img


def _draw_central_circle(K, to_device_from_world, img):
    """Draw central circle on the image"""

    circle_radius = 9.15

    res = 25
    circle_points_projected = np.zeros((res, 2), dtype=np.int32)
    for i in range(res):
        angle = i / res * np.pi * 2
        circle_points_world = (
            np.array([np.cos(angle), 0, np.sin(angle)]) * circle_radius
        )
        circle_points_projected[i] = project_to_screen(
            K, to_device_from_world, circle_points_world
        )

    img = cv2.polylines(
        img, [circle_points_projected], isClosed=True, color=(0, 165, 255), thickness=3
    )

    return img


def _draw_middle_line(K, to_device_from_world, img):
    """Draw middle/main line"""

    img = _project_and_draw_lines(
        K,
        to_device_from_world,
        [
            back_middle_line_world,
            front_middle_line_world,
        ],
        img,
    )

    return img


def _draw_border_lines(K, to_device_from_world, img):
    """Draw border lines"""

    img = _project_and_draw_lines(
        K,
        to_device_from_world,
        [
            corner_back_left_world,
            corner_front_left_world,
            corner_front_right_world,
            corner_back_right_world,
        ],
        img,
    )

    return img


def _draw_penalty_areas(K, to_device_from_world, img):
    """
    Draw penalty areas on both sides
    """

    penalty_left_front_goal_world = [-52.5, 0, -20.16]
    penalty_left_front_field_world = [-36, 0, -20.16]
    penalty_left_back_field_world = [-36, 0, 20.16]
    penalty_left_back_goal_world = [-52.5, 0, 20.16]

    img = _project_and_draw_lines(
        K,
        to_device_from_world,
        [
            penalty_left_front_goal_world,
            penalty_left_front_field_world,
            penalty_left_back_field_world,
            penalty_left_back_goal_world,
        ],
        img,
    )

    penalty_right_front_goal_world = [52.5, 0, -20.16]
    penalty_right_front_field_world = [36, 0, -20.16]
    penalty_right_back_field_world = [36, 0, 20.16]
    penalty_right_back_goal_world = [52.5, 0, 20.16]

    img = _project_and_draw_lines(
        K,
        to_device_from_world,
        [
            penalty_right_front_goal_world,
            penalty_right_front_field_world,
            penalty_right_back_field_world,
            penalty_right_back_goal_world,
        ],
        img,
    )

    return img


class KeyPoints:
    """
    Class of key points to be used to solve the PnP
    and estimate the focal length
    """

    def __init__(self):
        self.right_circle = None
        self.left_circle = None
        self.behind_circle = None
        self.front_circle = None
        self.front_middle_line = None
        self.back_middle_line = None
        self.corner_back_left = None
        self.corner_back_right = None
        self.corner_front_left = None
        self.corner_front_right = None

        
    def compute_points_array(self):
        points_list = [self.right_circle, # Verified
                        self.left_circle, # Verified
                        self.behind_circle, # Verified
                        self.front_circle, # Verified
                        self.back_middle_line, # Corrected & verified
                        self.front_middle_line,# Corrected & verified 
                        self.corner_back_left, 
                        self.corner_back_right, 
                        self.corner_front_left, 
                        self.corner_front_right]
        
        for i, pnt in enumerate(points_list):
            if pnt is None:
                pnt = [float('nan'), float('nan')]
            
            pnt = list(pnt)
            points_list[i] = pnt
        
        points_array = np.array(points_list, dtype=float)
        
        return points_array

    def draw(self, img):
        """Draw all key points on image"""
        img = draw_point(img, self.right_circle)
        img = draw_point(img, self.left_circle)
        img = draw_point(img, self.behind_circle)
        img = draw_point(img, self.front_circle)
        img = draw_point(img, self.front_middle_line)
        img = draw_point(img, self.back_middle_line)
        img = draw_point(img, self.corner_back_left)
        img = draw_point(img, self.corner_back_right)
        img = draw_point(img, self.corner_front_left)
        img = draw_point(img, self.corner_front_right)

        return img

    def __str__(self):
        return (
            f"Right circle: {self.right_circle}\nLeft circle: {self.left_circle}\n"
            f"Behing circle: {self.behind_circle}\nFront circle: {self.front_circle}\n"
            f"Back middle line: {self.back_middle_line}\n"
            f"Front middle line: {self.front_middle_line}\n"
            f"Corner back left: {self.corner_back_left}\n"
            f"Corner back right: {self.corner_back_right}\n"
            f"Corner front left: {self.corner_front_left}\n"
            f"Corner front right: {self.corner_front_right}\n"
        )

    def make_2d_3d_association_list(self):
        """
        Define set of pixels (2D) and its correspondance of points in world (3D)
        to feed the PnP solver.
        """
        pixels = []
        points_world = []

        if self.right_circle is not None:
            pixels.append(self.right_circle)
            points_world.append(right_circle_world)
        if self.left_circle is not None:
            pixels.append(self.left_circle)
            points_world.append(left_circle_world)
        if self.behind_circle is not None:
            pixels.append(self.behind_circle)
            points_world.append(behind_circle_world)
        if self.front_circle is not None:
            pixels.append(self.front_circle)
            points_world.append(front_circle_world)
        if self.front_middle_line is not None:
            pixels.append(self.front_middle_line)
            points_world.append(front_middle_line_world)
        if self.back_middle_line is not None:
            pixels.append(self.back_middle_line)
            points_world.append(back_middle_line_world)
        if self.corner_front_left is not None:
            pixels.append(self.corner_front_left)
            points_world.append(corner_front_left_world)
        if self.corner_front_right is not None:
            pixels.append(self.corner_front_right)
            points_world.append(corner_front_right_world)
        if self.corner_back_left is not None:
            pixels.append(self.corner_back_left)
            points_world.append(corner_back_left_world)
        if self.corner_back_right is not None:
            pixels.append(self.corner_back_right)
            points_world.append(corner_back_right_world)

        pixels = np.array(pixels, dtype=np.float32)
        points_world = np.array(points_world)

        return pixels, points_world

    def compute_focal_length(self, guess_fx):
        """
        Compute the focal length based on the central circle.
        If we cant spot the central circle, we return the default incoming value
        """
        if self.right_circle is None and self.left_circle is None:
            return guess_fx

        if self.right_circle is not None and self.left_circle is not None:
            fx = (
                (self.right_circle[0] - self.left_circle[0])
                * DIST_TO_CENTER
                / (right_circle_world[0] - left_circle_world[0])
            )
            return fx

        if self.behind_circle is None or self.front_circle is None:
            return guess_fx

        central = [
            int((self.behind_circle[0] + self.front_circle[0]) / 2),
            int((self.behind_circle[1] + self.front_circle[1]) / 2),
        ]
        if self.right_circle is None:
            fx = (
                (central[0] - self.left_circle[0])
                * DIST_TO_CENTER
                / (-left_circle_world[0])
            )
            return fx

        if self.left_circle is None:
            fx = (
                (self.right_circle[0] - central[0])
                * DIST_TO_CENTER
                / (right_circle_world[0])
            )
            return fx

        return fx
    
    def find_closer_point_on_line(self, point, line):
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
    
    def extend_key_points_set(self, K, to_device_from_world, key_lines):
        """
        As the PnP is not so performant with only a few points, we try to get closer
        to the Perspective-n-Line algo by projecting corners even if they are not visible

        We modifiy the key_points set.
        """

        if self.corner_back_right is None and self.corner_back_left is None:
            pt = project_to_screen(K, to_device_from_world, corner_front_right_world)
            self.corner_front_right = self.find_closer_point_on_line(
                pt, key_lines.front_line
            )
            pt = project_to_screen(K, to_device_from_world, corner_front_left_world)
            self.corner_front_left = self.find_closer_point_on_line(
                pt, key_lines.front_line
            )
            pt = project_to_screen(K, to_device_from_world, corner_back_right_world)
            self.corner_back_right = self.find_closer_point_on_line(
                pt, key_lines.back_line
            )
            pt = project_to_screen(K, to_device_from_world, corner_back_left_world)
            self.corner_back_left = self.find_closer_point_on_line(pt, key_lines.back_line)

        if (
            self.corner_back_right is not None
            and key_lines.right_goal_line is not None
        ):
            self.corner_front_right = intersect(
                key_lines.right_goal_line, key_lines.front_line
            )
        if self.corner_back_left is not None and key_lines.left_goal_line is not None:
            self.corner_front_left = intersect(
                key_lines.left_goal_line, key_lines.front_line
            )
