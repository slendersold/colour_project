import numpy as np
import cv2

class RectangleDetector:
    def __init__(self, image, show_image=False):
        """
        Initialize the RectangleDetector with an image and a flag to show the image.

        Parameters:
        image (numpy array): The input image in which to detect and draw rectangles.
        show_image (bool): Flag to indicate if the image with drawn rectangles should be shown.
        """
        self.image = image  # Ensure the image is a numpy array!
        self.show_image = show_image  # Store the flag to determine if the processed image should be shown
        self.coordinates = {}  # Dictionary to store detected rectangles and their coordinates
        self.processed_image = None  # Variable to store the image with drawn rectangles if show_image is True

    def preprocessing(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to the grayscale image
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        
        # Apply binary thresholding to the blurred image
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

        return(thresh)


    def detect_coordinates(self):
        """
        Detect light rectangular contours in the image and draw them on the image if specified.

        Returns:
        tuple: A tuple containing:
               - dict: A dictionary of detected rectangles with their coordinates.
               - numpy array: The image with detected rectangles drawn on it (if show_image is True).
        """


        # 1. PREPROCESSING

        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        # 2. VERTICES AND AREAS DETECTION
        vertices = []  # List to store vertices of detected rectangles
        areas = []  # List to store areas of detected rectangles
        for contour in contours:
            # Approximate the contour to a polygon
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(approx)
            
            # Check if the polygon has 4 vertices and area greater than 1000 to consider it a rectangle
            if len(approx) == 4 and area > 1000:
                vertices.append(approx)
                areas.append(area)

        # Convert the vertices to (y0, y1, x0, x1) format
        vertices_new_format = [(vert[:, 0, 1].min(), vert[:, 0, 1].max(), vert[:, 0, 0].min(), vert[:, 0, 0].max()) for vert in vertices]

        # Combine areas and vertices for sorting
        combined = list(zip(areas, vertices_new_format))
        
        # Sort the combined list by area in descending order
        combined.sort(key=lambda x: x[0], reverse=True)
        
        # Define keys for the rectangles
        keys = ["rect_CA", "rect_1000", "rect_750", "rect_500", "rect_400", "rect_300"]
        
        # Populate the coordinates with sorted rectangles
        for i, key in enumerate(keys):
            if i < len(combined):
                area, (y0, y1, x0, x1) = combined[i]
                self.coordinates[key] = {
                    'y0': y0,
                    'y1': y1,
                    'x0': x0,
                    'x1': x1
                }


        # 3. SHOW PROCESSED IMAGE (INPUT IMAGE WITH DETECTED RECTANGLES)
        # If show_image flag is True, draw the rectangles on the image
        if self.show_image:
            self.processed_image = self.image.copy()
            for rect in vertices:
                cv2.drawContours(self.processed_image, [rect], -1, (0, 255, 0), 20)
            return self.coordinates, self.processed_image
        else:
            return self.coordinates, None

# Example usage:
# image = cv2.imread('path_to_image')
# detector = RectangleDetector(image, show_image=True)
# image_processed = detector.processing()
# coordinates = detector.detect_coordinates(image_processed, show_image = False)
