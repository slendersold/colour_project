import numpy as np
import cv2


class CircleDetector:
    def __init__(self, image, rect_name=""):
        """
        Initialize the CircleDetector with an image and a rectangle name.

        Parameters:
        image (numpy array): The input image in which to detect and draw circles.
        rect_name (str): The name used for rectangle identification in the dictionary keys.
        """
        self.image = np.array(image)  # Ensure the image is a numpy array
        self.rect_name = rect_name  # Store the rectangle name for dictionary keys

    def preprocessing_sobel(self, image):
        """
        Compute the Sobel magnitude of the input image.

        Parameters:
        image (numpy array): The input image.

        Returns:
        numpy array: The Sobel magnitude image.
        """
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        image_processed = np.sqrt(sobelx**2 + sobely**2)
        image_processed = cv2.convertScaleAbs(image_processed)
        return image_processed

    def preprocessing_clahe(self, image):
        """
        Enhance the contrast of the image using CLAHE.

        Parameters:
        image (numpy array): The input image.

        Returns:
        numpy array: The image with enhanced contrast.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        image_processed = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return image_processed

    def preprocessing_general(self, image):
        """
        Preprocess the image to convert to grayscale, apply Gaussian blur, and detect edges.

        Parameters:
        image (numpy array): The input image.

        Returns:
        numpy array: The edges detected in the image.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        image_processed = cv2.Canny(blurred, 50, 150)

        return image_processed

    def detect_coordinates(self, image, averaging_threshold):
        """
        Detect circular contours and their coordinates from the edge-detected image.

        Parameters:
        image (numpy array): The edge-detected image.

        Returns:
        list: A list of detected circles with their coordinates and radii.
        """

        def is_circle(self, contour, width, tolerance=0.2):

            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            if perimeter < int(width / 6):
                return False
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            return 1 - tolerance <= circularity <= 1 + tolerance

        def average(self, coordinates, threshold=10):

            def distance(coord1, coord2):
                return np.linalg.norm(np.array(coord1) - np.array(coord2))

            unique_groups = []
            for coord in coordinates:
                placed = False
                for group in unique_groups:
                    if all(distance(coord, member) <= threshold for member in group):
                        group.append(coord)
                        placed = True
                        break
                if not placed:
                    unique_groups.append([coord])

            mean_values = [
                np.mean(group, axis=0).astype(int).tolist() for group in unique_groups
            ]
            return mean_values

        # Find contours in the edge-detected image
        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours to find circles
        circles = [contour for contour in contours if is_circle(contour, self.width)]

        # Write circle coordinates
        circles_coord = []
        for contour in circles:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            circles_coord.append([center[0], center[1], radius])

        mean_radius = int(np.mean([c[2] for c in circles_coord]))  #
        circles_coord = average(circles_coord, threshold=averaging_threshold)  #
        for contour in circles_coord:  #
            contour[2] = mean_radius  #

        return circles_coord

    def circ_to_dict(self, circles_coord):
        """
        Convert circle coordinates to a dictionary with specific keys.

        Parameters:
        circles_coord (list): List of circles with their coordinates and radii.
        rect_name (str): The name used for generating keys in the dictionary.

        Returns:
        dict: A dictionary with circle coordinates and radii.
        """
        # Sort circles by y-coordinate
        sorted_by_y = sorted(circles_coord, key=lambda item: item[1])

        # Sort each subgroup of 4 circles by x-coordinate
        n = 4
        for i in range(0, len(sorted_by_y), n):
            subgroup = sorted_by_y[i : i + n]
            sorted_subgroup = sorted(subgroup, key=lambda item: item[0])
            sorted_by_y[i : i + n] = sorted_subgroup

        # Characters and numbers for generating keys
        characters = ["A", "B", "C", "D"]
        numbers = [6, 5, 4, 3, 2, 1]

        result_dict = {}
        index = 0

        # Generate keys and fill the dictionary with circle coordinates
        for num in numbers:
            for char in characters:
                if index < len(sorted_by_y):
                    key = f"{char}{num}{self.rect_name}"
                    x, y, radius = sorted_by_y[index]
                    result_dict[key] = {
                        "x_centroid": x,
                        "y_centroid": y,
                        "radius": radius,
                    }
                    index += 1

        return result_dict


# Example usage:
image = cv2.imread("path_to_image")
detector = CircleDetector(image, rect_name="example")
sobel_processed_image = detector.preprocessing_sobel()
clahe_processed_image = detector.preprocessing_clahe(sobel_processed_image)
general_processed_image = detector.preprocessing_general(clahe_processed_image)
coordinates = detector.detect_coordinates(general_processed_image)
coordinates = detector.coordinates_to_dict(coordinates)
