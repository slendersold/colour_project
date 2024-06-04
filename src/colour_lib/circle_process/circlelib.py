import numpy as np
import matplotlib.pyplot as plt
from colour import XYZ_to_Lab, delta_E

class CircleColors:
    def __init__(self, number_of_circles):
        self.number_of_circles = number_of_circles
        return

    def apply_gammaCorr(img, gammas):
        # Transpose to set channel as 1st for iter
        _transposed = np.transpose(img, (2, 0, 1))

        # Perform inverse gamma correction
        ret = np.empty_like(_transposed)
        for i, gamma in enumerate(gammas):
            ret[i] = _transposed[i] ** (gamma)

        # Transpose back to original shape
        ret = np.transpose(ret, (1, 2, 0))
        return ret

    def calculate_delta_E (self, observe, reference):
        observe = XYZ_to_Lab(observe)
        reference = XYZ_to_Lab(reference)
        deltas = np.zeros((observe.shape[0], 1))
        for i in range(observe.shape[0]):
            a = reference[i, :]
            b = observe[i, :]
            deltas[i] = delta_E(a, b, method="CIE 2000")
        return deltas
    
    def draw_circle_mask(image, coord, radius):
        mask_shape = image.shape
        coords_y, coords_x = np.ogrid[: mask_shape[0], : mask_shape[1]]
        circle_mask = np.zeros(mask_shape[:2], dtype=bool)
        for idx, centroids in coord.iterrows():
            centroid_y, centroid_x = centroids["Y"], centroids["X"]
            circle_mask = circle_mask | ((coords_y - centroid_y) ** 2 + (coords_x - centroid_x) ** 2 <= radius**2)

        plt.imshow(image)
        plt.imshow(circle_mask, alpha=0.5)

    def create_circle_mask(image, centroids, radius):
        mask_shape = image.shape[:2]
        coords_y, coords_x = np.ogrid[: mask_shape[0], : mask_shape[1]]
        circle_mask = (coords_y - centroids["Y"])**2 + (coords_x - centroids["X"])**2 <= radius**2
        circle_mask = np.expand_dims(circle_mask, axis=-1)  # Add a channel dimension
        circle_mask = np.tile(circle_mask, (1, 1, image.shape[2]))  # Tile to match image channels
        return circle_mask

    def calc_rectangle(self, image, coord, radius):
        #create an array of mean values of one set of circles from A1 to D6
        mean_values = []
        for idx, centroids in coord.iterrows():
            circle = self.create_circle_mask(image, centroids, radius)
            masked_image = image * circle
            mean_in_circle = []
            for i in range(image.shape[2]):
                channel_values = masked_image[:,:,i][masked_image[:,:,i] != 0]
                mean_in_circle.append(channel_values.mean())
            mean_in_circle = np.nan_to_num(mean_in_circle)  # Replace NaN with 0
            mean_values.append(mean_in_circle.tolist())
        return np.array(mean_values)
    
    def calc_rectangles(self, image, coords, rads, zones):
        #calculates mean value of all mean values from A1 to D6 in one single image
        mean_values = np.zeros((self.number_of_circles,image.shape[2]))
        for idx, v in enumerate(zones):
            mean_values = mean_values + self.calc_rectangle(image, coords[v], rads[v])
        mean_values = mean_values/(idx+1)
        return mean_values