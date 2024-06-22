from sklearn import linear_model
from ..regressors.AbstractRegressor import AbstractRegressor


class LassoRegressor(AbstractRegressor):
    def __init__(self, train_data, reference_data, alpha):
        self.reg = linear_model.Lasso(alpha=alpha)
        self.reg.fit(train_data, reference_data)

    def predict(self, img):
        return super().predict(img=img)
