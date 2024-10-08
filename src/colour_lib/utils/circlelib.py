import numpy as np
import matplotlib.pyplot as plt


def draw_circle_mask(image, coord, radius):
    mask_shape = image.shape
    coords_y, coords_x = np.ogrid[: mask_shape[0], : mask_shape[1]]
    circle_mask = np.zeros(mask_shape[:2], dtype=bool)
    for idx, centroids in coord.iterrows():
        centroid_y, centroid_x = centroids["Y"], centroids["X"]
        circle_mask = circle_mask | (
            (coords_y - centroid_y) ** 2 + (coords_x - centroid_x) ** 2 <= radius**2
        )
    plt.imshow(image)
    plt.imshow(circle_mask, alpha=0.5)


def create_circle_mask(image, centroids, radius):
    mask_shape = image.shape[:2]
    coords_y, coords_x = np.ogrid[: mask_shape[0], : mask_shape[1]]
    circle_mask = (coords_y - centroids["Y"]) ** 2 + (
        coords_x - centroids["X"]
    ) ** 2 <= radius**2
    circle_mask = np.expand_dims(circle_mask, axis=-1)  # Add a channel dimension
    circle_mask = np.tile(
        circle_mask, (1, 1, image.shape[2])
    )  # Tile to match image channels
    return circle_mask


def calc_rectangle(image, coord, radius):
    # create an array of mean values of one set of circles from A1 to D6
    mean_values = []
    for idx, centroids in coord.iterrows():
        circle = create_circle_mask(image, centroids, radius)
        masked_image = image * circle
        mean_in_circle = []
        for i in range(image.shape[2]):
            channel_values = masked_image[:, :, i][masked_image[:, :, i] != 0]
            mean_in_circle.append(channel_values.mean())
        mean_in_circle = np.nan_to_num(mean_in_circle)  # Replace NaN with 0
        mean_values.append(mean_in_circle.tolist())
    return np.array(mean_values)


def calc_rectangles(image, coords, rads, zones):
    # calculates mean value of all mean values from A1 to D6 in one single image
    mean_values = np.zeros((list(coords.items())[1][1].shape[0], image.shape[2]))
    for idx, v in enumerate(zones):
        mean_values = mean_values + calc_rectangle(image, coords[v], rads[v])
    mean_values = mean_values / (idx + 1)
    return mean_values


def calc_slide(mnval_CA, mnval_slide):
    # appends clear area colors value to the beginning of data array
    # thus formating in standart form: [CA, A1, ..., D6].
    full_table = np.append(mnval_CA, mnval_slide, axis=0)

    return full_table
