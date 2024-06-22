import numpy as np
import matplotlib.pyplot as plt


class ObjectColor:
    def __init__(self):
        pass

    def calculate_rgb(self, coordinates, zarr, filter="_1000"):

        def mean_rgb_circ(image, x, y, r, margin=0.3):

            r = int(r * margin)

            Y, X = np.ogrid[: image.shape[0], : image.shape[1]]

            dist_from_center = np.sqrt((X - x) ** 2 + (Y - y) ** 2)

            mask = dist_from_center <= r

            mean_color = []
            for i in range(image.shape[2]):
                channel_mean = image[:, :, i][mask].mean()
                mean_color.append(channel_mean)

            return np.array(mean_color)

        def mean_rgb_rect(image, y0, y1, x0, x1, margin=0.3):

            y0, y1 = sorted([y0, y1])
            x0, x1 = sorted([x0, x1])

            h = y1 - y0
            w = x1 - x0

            y0_m, y1_m = int(y0 + h * margin), int(y1 - h * margin)
            x0_m, x1_m = int(x0 + w * margin), int(x1 - w * margin)

            region = image[y0_m:y1_m, x0_m:x1_m]
            mean_color = region.mean(axis=(0, 1))

            return np.array(mean_color)

        coordinates_filtered = {
            key: value
            for key, value in coordinates.items()
            if (filter in key) and ("rect" not in key)
        }
        coordinates_filtered["rect_CA"] = coordinates["rect_CA"]
        coordinates_filtered["rect_dark"] = coordinates["rect_dark"]

        obj_rgb = {}
        for i, (key, values) in enumerate(coordinates_filtered.items()):
            print(key)
            if key == "rect_CA" or key == "rect_dark":
                x0 = values["x0"]
                x1 = values["x1"]
                y0 = values["y0"]
                y1 = values["y1"]

                mean_color = mean_rgb_rect(zarr, y0, y1, x0, x1, margin=0.3)
                obj_rgb[key] = {
                    "r": mean_color[0],
                    "g": mean_color[1],
                    "b": mean_color[2],
                }

            else:
                x = values["x_centroid"]
                y = values["y_centroid"]
                r = values["radius"]

                mean_color = mean_rgb_circ(zarr, x, y, r, margin=0.3)
                obj_rgb[key] = {
                    "r": mean_color[0],
                    "g": mean_color[1],
                    "b": mean_color[2],
                }

        return obj_rgb

    def plot_palette(self, obj_rgb):
        """
        Plot a color grid based on the mean RGB values stored in the obj_rgb dictionary.

        Args:
        obj_rgb (dict): Dictionary containing mean RGB values for each object.
        """
        fig, ax = plt.subplots(figsize=(4, 7))

        # Create a grid of colors based on the obj_rgb keys
        color_grid = np.zeros(
            (7, 4, 3)
        )  # Assuming 7 rows and 4 columns for CA and A-D with 1-6

        # Fill the color grid with the mean colors from the obj_rgb
        for key, color in obj_rgb.items():
            if key == "rect_CA":
                for i in range(2):
                    color_grid[0, i, :] = [color["r"], color["g"], color["b"]]
            elif key == "rect_dark":
                for i in range(2):
                    color_grid[0, 2 + i, :] = [color["r"], color["g"], color["b"]]
            else:
                row_label = key[0]  # Extract the row label (A, B, C, ...)
                col_label = key[1]  # Extract the column label (6, 5, 4, ...)
                row_idx = 7 - int(
                    col_label
                )  # Calculate the row index (6 maps to 0, 5 maps to 1, etc.)
                col_idx = ord(row_label) - ord(
                    "A"
                )  # Calculate the column index (A maps to 0, B maps to 1, etc.)
                color_grid[row_idx, col_idx, :] = [color["r"], color["g"], color["b"]]

        # Normalize the color values to [0, 1] for displaying with imshow
        color_grid /= 255.0

        # Display the grid
        ax.imshow(color_grid, aspect="auto")

        # Set the tick labels
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(["A", "B", "C", "D"])
        ax.set_yticks(np.arange(7))
        ax.set_yticklabels(["CA/B", "6", "5", "4", "3", "2", "1"])

        # plt.show()

        # Render the figure canvas and convert it to a NumPy array
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Close the plot
        plt.close(fig)

        return img_array
