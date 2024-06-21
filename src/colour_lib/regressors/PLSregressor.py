from sklearn.cross_decomposition import PLSRegression


class PLSRegressor:
    def __init__(self, train_data, reference_data):
        self.reg = PLSRegression(n_components=train_data.shape[-1])
        self.reg.fit(train_data, reference_data)

    def predict(self, img):
        mod_img = img.copy()

        for strip in mod_img:
            strip = self.reg.predict(strip)

        # return np.clip(mod_img, 0, 1)
        return mod_img
