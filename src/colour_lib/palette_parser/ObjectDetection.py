import numpy as np
import cv2
import matplotlib.pyplot as plt


class ObjectDetection:
    def __init__(self):
        pass

    def find_rectangles(self, image_to_find, image_to_show, show_image=True):

        # 1. FIND CONTOURS
        contours, _ = cv2.findContours(
            image_to_find, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 2. VERTICES AND AREAS DETECTION
        vertices = []  # List to store vertices of detected rectangles
        areas = []  # List to store areas of detected rectangles
        for contour in contours:
            # Approximate the contour to a polygon
            approx = cv2.approxPolyDP(
                contour, 0.02 * cv2.arcLength(contour, True), True
            )
            area = cv2.contourArea(approx)

            # Check if the polygon has 4 vertices and area greater than 1000 to consider it a rectangle
            if len(approx) == 4 and area > 20000:
                vertices.append(approx)
                areas.append(area)

        # Convert the vertices to (y0, y1, x0, x1) format
        vertices_new_format = [
            (
                vert[:, 0, 1].min(),
                vert[:, 0, 1].max(),
                vert[:, 0, 0].min(),
                vert[:, 0, 0].max(),
            )
            for vert in vertices
        ]

        # Combine areas and vertices for sorting
        combined = list(zip(areas, vertices_new_format))

        # Sort the combined list by area in descending order
        combined.sort(key=lambda x: x[0], reverse=True)

        # Define keys for the rectangles
        keys = ["rect_CA", "rect_1000", "rect_750", "rect_500"] #, "rect_400", "rect_300"]

        # Populate the coordinates with sorted rectangles
        coordinates = {}
        for i, key in enumerate(keys):
            if i < len(combined):
                area, (y0, y1, x0, x1) = combined[i]
                coordinates[key] = {"y0": y0, "y1": y1, "x0": x0, "x1": x1}
        
        # add dark area
        CA_h = coordinates['rect_CA']['y1'] - coordinates['rect_CA']['y0']
        coordinates['rect_dark'] = coordinates['rect_CA'].copy()
        coordinates['rect_dark']['y0'] = int(coordinates['rect_CA']['y1'] + 0.3*CA_h)
        coordinates['rect_dark']['y1'] = int(coordinates['rect_CA']['y1'] + 0.6*CA_h)

        # 3. SHOW INPUT IMAGE WITH DETECTED RECTANGLES (OPTIONAL)

        if show_image:
            image_to_show_ = image_to_show.copy()
            for rect in vertices:
                cv2.drawContours(image_to_show_, [rect], -1, (250, 199, 16), 20)
            return coordinates, image_to_show_
        else:
            return coordinates, None

    def find_circles(self, image, averaging_threshold, tolerance = 0.2):

        def is_circle(contour, width, tolerance):

            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            if perimeter < int(width / 6):
                return False
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            return 1 - tolerance <= circularity <= 1 + tolerance

        def average(coordinates, threshold=10):

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
        width = image.shape[1]
        circles = [
            contour
            for contour in contours
            if is_circle(contour=contour, width=width, tolerance=tolerance)
        ]

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

    def circles_coordinates_as_dictionary(self, circles_coord, rect_name):

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
                    key = f"{char}{num}{rect_name}"
                    x, y, radius = sorted_by_y[index]
                    result_dict[key] = {
                        "x_centroid": x,
                        "y_centroid": y,
                        "radius": radius,
                    }
                    index += 1

        return result_dict

    def draw_rectangles(self, image, rect_coord_light):
        for key, value in rect_coord_light.items():
            y0, y1, x0, x1 = value["y0"], value["y1"], value["x0"], value["x1"]
            contour = np.array(
                [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32
            ).reshape((-1, 1, 2))
            contours = [contour]
            cv2.drawContours(image, contours, -1, (0, 255, 0), 20)

    def draw_circles(self, image, circle_coord_light):
        for key, value in circle_coord_light.items():
            x, y, radius = value["x_centroid"], value["y_centroid"], value["radius"]
            center = (int(x), int(y))
            cv2.circle(image, center, radius, (0, 255, 0), 20)

    def scale(self, zarr_scaled, zarr_light, obj_coord_light):
        scale_factor = round(zarr_scaled.shape[0] / zarr_light.shape[0])

        obj_coord_scaled = {
            outer_key: {
                inner_key: int(value * scale_factor)
                for inner_key, value in inner_dict.items()
            }
            for outer_key, inner_dict in obj_coord_light.items()
        }

        return obj_coord_scaled
