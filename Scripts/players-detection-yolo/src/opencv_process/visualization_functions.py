

import cv2
import numpy as np

def draw_detected_objects(original_frame, detected_objects, circle_color):
    frame = original_frame.copy()
    
    # Fixed dimensions for the triangle
    tri_height = 25
    tri_base_half_length = 15
    vertical_offset = 20
    
    for detected_obj in detected_objects:
        x1, y1, x2, y2 = detected_obj.bbox
        
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        
        width, height = (x2 - x1), (y2 - y1)
        center = (x1 + width/2, y1 + height/2)
        
        box_id = detected_obj.id
        # box_id = 0
        
        # Calculate the bottom center of the bounding box for the ellipse
        bottom_center = (int(center[0]), int(center[1] + height / 2))
        
        # Define the axes for the ellipse and angle
        ellipse_axes = (int(width / 2), int(height / 10))
        ellipse_angle = 0
        ellipse_thickness = 4

        # Draw the bottom half ellipse in blue with the specified thickness
        cv2.ellipse(frame, bottom_center, ellipse_axes, ellipse_angle, 0, 180, circle_color, ellipse_thickness)

        # Calculate the bottom point of the triangle (above the bounding box)
        top_point_triangle = (int(center[0]), int(center[1] - height / 2) - vertical_offset)

        # Triangle points
        p1 = (top_point_triangle[0], top_point_triangle[1] + tri_height)
        p2 = (top_point_triangle[0] - tri_base_half_length, top_point_triangle[1])
        p3 = (top_point_triangle[0] + tri_base_half_length, top_point_triangle[1])

        # Draw the filled triangle in white for the ID
        cv2.drawContours(frame, [np.array([p1, p2, p3])], 0, (255, 255, 255), -1)

        # Add the ID text in black, centered in the triangle
        text_size = cv2.getTextSize(str(box_id), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = p1[0] - text_size[0] // 2
        text_y = p1[1] - 2* text_size[1] // 3
        cv2.putText(frame, str(box_id), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return frame

if __name__ == "__main__":
    pass