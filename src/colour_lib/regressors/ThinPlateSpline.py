from ..regressors.AbstractRegressor import AbstractRegressor
from tps import ThinPlateSpline
import numpy as np


class TPSRegressor(AbstractRegressor):
    def __init__(self, train_data, reference_data, **kwargs):
        # Create the tps object
        self.reg = ThinPlateSpline(alpha=kwargs["alpha"])  # 0 Regularization

        # Fit the control and target points
        self.reg.fit(reference_data, train_data)

    def predict(self, img):
        mod_img = img.copy()

        for i, strip in enumerate(mod_img):
            strip = self.reg.transform(strip)
            mod_img[i] = strip

        return np.clip(mod_img, 0, 1)
