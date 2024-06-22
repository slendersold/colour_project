import numpy as np
import cv2


class ImageProcessing:
    def __init__(self):
        pass

    def gray_thresh(self, image, scanner_type):

        if scanner_type == "huron":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 0)
            _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)  # 200
            return thresh

        if scanner_type == "polaris":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2
            )
            return thresh

    def sobel(self, image):
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

    def clahe(self, image):
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

    def gray_blur_canny(self, image):
        """
        Preprocess the image to convert to grayscale, apply Gaussian blur, and detect edges.

        Parameters:
        image (numpy array): The input image.

        Returns:
        numpy array: The edges detected in the image.
        """

        def auto_canny(image, sigma=0.5):
            v = np.median(image)
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            return cv2.Canny(image, lower, upper)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)  # (9, 9)
        image_processed = cv2.Canny(blurred, 50, 150)

        return image_processed

    def get_brightest_color(self, image):
        image_array = np.array(image)
        brightest_color = np.amax(image_array, axis=(0, 1))
        return tuple(map(int, brightest_color))

    def add_bottom_border(self, image, color, border_size=40):
        color_bgr = (color[2], color[1], color[0])

        bordered_image = cv2.copyMakeBorder(
            image,
            top=0,
            bottom=border_size,
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=color_bgr,
        )

        return bordered_image

    def increase_contrast(self, image, alpha=1.5, beta=0):
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted
