
import cv2
import numpy as np
import os

class InteractiveImageMarker:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if self.original_image is None:
            raise ValueError("Error loading image")
        self.image = self.original_image.copy()
        self.points = []  # To store points clicked by the user
        self.prompts = [
            "Right circle",
            "Left circle",
            "Behind circle",
            "Front circle",
            "Back middle line",
            "Front middle line",
            "Corner back left",
            "Corner back right",
            "Corner front left",
            "Corner front right"
        ]
    
    def click_and_draw(self, event, x, y, flags, param):
        """Callback function for mouse click events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Draw a blue point with an index order number next to it
            cv2.circle(self.image, (x, y), 3, (255, 0, 0), -1)
            index = len(self.points)
            cv2.putText(self.image, str(index), (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            self.points.append((x, y))
            self.update_prompt_display()
            cv2.imshow("Image", self.image)
    
    def update_prompt_display(self):
        """Update the image with the current prompt for the next point to be marked."""
        self.image = self.original_image.copy()
        for i, (x, y) in enumerate(self.points):
            cv2.circle(self.image, (x, y), 3, (255, 0, 0), -1)
            cv2.putText(self.image, str(i), (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Display the prompt for the next point if there are prompts left
        if len(self.points) < len(self.prompts):
            prompt = self.prompts[len(self.points)]
            cv2.putText(self.image, prompt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Image", self.image)
    
    def redraw_points(self):
        """Redraw all points on the image and update the prompt."""
        self.image = self.original_image.copy()
        for i, (x, y) in enumerate(self.points):
            cv2.circle(self.image, (x, y), 3, (255, 0, 0), -1)
            cv2.putText(self.image, str(i), (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        self.update_prompt_display()
    
    def remove_last_point(self):
        """Remove the last point added by the user."""
        if self.points:
            self.points.pop()
            self.redraw_points()
    
    def save_points(self):
        """Save the points to a file in the same directory as the input image."""
        points_array = np.array(self.points)
        print("Points Array:\n", points_array)
        
        # Extract directory from the image path
        directory = os.path.dirname(self.image_path)
        
        # Construct the file path for saving the points
        base_filename = os.path.basename(self.image_path)
        name, ext = os.path.splitext(base_filename)
        points_file_path = os.path.join(directory, f"{name}_points.npy")
        
        np.save(points_file_path, points_array)
        print(f"Points saved to {points_file_path}")
    
    def run(self):
        """Main loop for handling events."""
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.click_and_draw)
        self.update_prompt_display()  # Initial prompt display
        cv2.imshow("Image", self.image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):
                self.remove_last_point()
            elif key == ord('s'):
                self.save_points()
                break
            elif key == 27:  # ESC key to exit
                break
        
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    marker = InteractiveImageMarker("../../../Datasets/soccer field layout/soccer_field_layout.png")
    marker.run()
