import numpy as np
from tps import ThinPlateSpline
from sklearn.preprocessing import StandardScaler

class TPSRegressor:
    def __init__(self, train_data, reference_data, alpha):
        # Create the tps object
        self.reg = ThinPlateSpline(alpha=alpha)  # 0 Regularization

        # Fit the control and target points
        self.reg.fit(train_data, reference_data)

    def predict(self, img):
        mod_img = img.copy()

        for strip in mod_img:
            strip = self.reg.transform(strip)

        # return np.clip(mod_img, 0, 1)
        return mod_img