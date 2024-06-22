import numpy as np
from abc import ABC, abstractmethod


class AbstractRegressor(ABC):
    @abstractmethod
    def __init__(self, train_data, reference_data, **kwargs):
        self.reg

    @abstractmethod
    def predict(self, img):
        mod_img = img.copy()

        for i, strip in enumerate(mod_img):
            strip = self.reg.predict(strip)
            mod_img[i] = strip

        return np.clip(mod_img, 0, 1)
