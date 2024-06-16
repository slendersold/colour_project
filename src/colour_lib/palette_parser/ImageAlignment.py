import cv2
import numpy as np


class ImageAlignment:
    def __init__(self):
        self.position = {
            "flip_horizontal": False,
            "flip_vertical": False,
            "flip_over": False,
            "rotation_angle": 0.0,
        }

    def flip(self, image_to_flip, pattern):

        y1000_more_than_yCA = False
        x1000_more_than_x500 = False

        if pattern["rect_1000"]["y0"] > pattern["rect_CA"]["y0"]:
            y1000_more_than_yCA = True
        if pattern["rect_1000"]["x0"] > pattern["rect_500"]["x0"]:
            x1000_more_than_x500 = True

        if y1000_more_than_yCA and x1000_more_than_x500:
            self.position["flip_horizontal"] = True
            flipped = cv2.flip(image_to_flip, 1)
        elif not y1000_more_than_yCA and not x1000_more_than_x500:
            self.position["flip_vertical"] = True
            flipped = cv2.flip(image_to_flip, 0)
        elif not y1000_more_than_yCA and x1000_more_than_x500:
            self.position["flip_over"] = True
            flipped = cv2.rotate(image_to_flip, cv2.ROTATE_180)
        else:
            flipped = image_to_flip

        return flipped

    def rotate(self, image_to_rotate, show_image, image_to_show):
        h, w = image_to_rotate.shape[:2]
        gray = cv2.cvtColor(image_to_rotate, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        theta_array = []
        for r_theta in lines:
            r, theta = np.array(r_theta[0], dtype=np.float64)
            # if normal is vertical (means that detected lines are horizontal)
            if (
                np.pi / 2 - 0.1 < theta < np.pi / 2 + 0.1
                or 3 * np.pi / 2 - 0.1 < theta < 3 * np.pi / 2 + 0.1
            ):
                theta_array.append(theta)
        theta_deg_mean = np.mean(np.rad2deg(theta_array))
        angle_to_rotate = 90 - theta_deg_mean
        self.position["rotation_angle"] = angle_to_rotate

        center = (w / 2, h / 2)
        matrix = cv2.getRotationMatrix2D(center, angle=-angle_to_rotate, scale=1.0)
        image_rotated = cv2.warpAffine(image_to_rotate, matrix, (w, h))

        if show_image:
            image_to_show = image_to_show.copy()
            for r_theta in lines:
                r, theta = np.array(r_theta[0], dtype=np.float64)

                # if normal is vertical (means that detected lines are horizontal)
                if (
                    np.pi / 2 - 0.1 < theta < np.pi / 2 + 0.1
                    or 3 * np.pi / 2 - 0.1 < theta < 3 * np.pi / 2 + 0.1
                ):
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * r
                    y0 = b * r
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(image_to_show, (x1, y1), (x2, y2), (0, 0, 255), 50)

        return image_rotated, image_to_show

    def give_original_coordinates(self, coordinates, w, h, type):

        def flip_coordinates_horizontal(coordinates, width, type):

            flipped_dict = {}

            if type == "circle":
                for key, value in coordinates.items():
                    x_new = width - value["x_centroid"]
                    flipped_dict[key] = {
                        "x_centroid": x_new,
                        "y_centroid": value["y_centroid"],
                        "radius": value["radius"],
                    }
            if type == "rectangle":
                for key, value in coordinates.items():
                    x0_new = width - value["x0"]
                    x1_new = width - value["x1"]
                    flipped_dict[key] = {
                        "y0": value["y0"],
                        "y1": value["y1"],
                        "x0": x0_new,
                        "x1": x1_new,
                    }

            return flipped_dict

        def flip_coordinates_vertical(coordinates, height, type):

            flipped_dict = {}

            if type == "circle":
                for key, value in coordinates.items():
                    y_new = height - value["y_centroid"]
                    flipped_dict[key] = {
                        "x_centroid": value["x_centroid"],
                        "y_centroid": y_new,
                        "radius": value["radius"],
                    }
            if type == "rectangle":
                for key, value in coordinates.items():
                    y0_new = height - value["y0"]
                    y1_new = height - value["y1"]
                    flipped_dict[key] = {
                        "y0": y0_new,
                        "y1": y1_new,
                        "x0": value["x0"],
                        "x1": value["x1"],
                    }

            return flipped_dict

        def rotate_coordinates(coordinates, width, height, theta, type):

            def rotate_coordinate(x, y, cx, cy, theta):

                # Convert angle to radians
                theta = np.radians(theta)
                # Step 1: Translate point to origin
                x_origin = x - cx
                y_origin = y - cy
                # Step 2: Apply rotation matrix
                x_rotated = x_origin * np.cos(theta) - y_origin * np.sin(theta)
                y_rotated = x_origin * np.sin(theta) + y_origin * np.cos(theta)
                # Step 3: Translate point back
                x_new = x_rotated + cx
                y_new = y_rotated + cy

                return x_new, y_new

            cx = int(width / 2)
            cy = int(height / 2)
            rotated_dict = {}

            if type == "circle":
                for key, value in coordinates.items():
                    x_new, y_new = rotate_coordinate(
                        value["x_centroid"], value["y_centroid"], cx, cy, theta
                    )
                    rotated_dict[key] = {
                        "x_centroid": x_new,
                        "y_centroid": y_new,
                        "radius": value["radius"],
                    }

            if type == "rectangle":
                for key, value in coordinates.items():
                    x0_new, y0_new = rotate_coordinate(
                        value["x0"], value["y0"], cx, cy, theta
                    )
                    x1_new, y1_new = rotate_coordinate(
                        value["x1"], value["y1"], cx, cy, theta
                    )
                    rotated_dict[key] = {
                        "y0": y0_new,
                        "y1": y1_new,
                        "x0": x0_new,
                        "x1": x1_new,
                    }
            return rotated_dict

        # MAIN PART
        coordinates_unsettled = coordinates.copy()

        # 1. FLIP

        if self.position.get("flip_horizontal"):
            print("Flip horizontal")
            coordinates_unsettled = flip_coordinates_horizontal(
                coordinates_unsettled, w, type
            )
        if self.position.get("flip_vertical"):
            print("Flip vertical")
            coordinates_unsettled = flip_coordinates_vertical(
                coordinates_unsettled, h, type
            )
        if self.position.get("flip_over"):
            print("Flip over")
            coordinates_unsettled = rotate_coordinates(
                coordinates_unsettled, w, h, 180, type
            )

        # 2. ROTATION

        rotation_angle = self.position.get("rotation_angle", 0)
        if rotation_angle != 0:
            print("Rotation angle:", rotation_angle)
            coordinates_unsettled = rotate_coordinates(
                coordinates_unsettled, w, h, rotation_angle, type
            )

        return coordinates_unsettled
