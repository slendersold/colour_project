from sklearn import linear_model


class LassoRegressor:
    def __init__(self, train_data, reference_data, alpha):
        self.reg = linear_model.Lasso(alpha=alpha)
        self.reg.fit(train_data, reference_data)

    def predict(self, img):
        mod_img = img.copy()

        for strip in mod_img:
            strip = self.reg.predict(strip)

        # return np.clip(mod_img, 0, 1)
        return mod_img
