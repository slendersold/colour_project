from sklearn.cross_decomposition import PLSRegression
from colour_lib.regressors.AbstractRegressor import AbstractRegressor


class PLSRegressor(AbstractRegressor):
    def __init__(self, train_data, reference_data):
        self.reg = PLSRegression(n_components=train_data.shape[-1])
        self.reg.fit(train_data, reference_data)

    def predict(self, img):
        return super().predict(img=img)
