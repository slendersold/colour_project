# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.cross_decomposition import PLSRegression
# from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


class VoteRegressor:
    def __init__(self, train_data, reference_data, random_state):
        # reg1 = GradientBoostingRegressor(random_state = random_state)
        # reg2 = RandomForestRegressor(random_state = random_state)
        # reg3 = PLSRegression()
        base_models = [
            ("linear", LinearRegression()),
            ("rf", RandomForestRegressor()),
            ("svr", SVR()),
        ]
        self.reg = VotingRegressor(estimators=base_models)
        self.reg = {
            "red reg": VotingRegressor(estimators=base_models),
            "green reg": VotingRegressor(estimators=base_models),
            "blue reg": VotingRegressor(estimators=base_models),
        }
        self.reg["red reg"].fit(
            train_data[:, 0].reshape(-1, 1), reference_data[:, 0].reshape(-1, 1)
        )
        self.reg["green reg"].fit(
            train_data[:, 1].reshape(-1, 1), reference_data[:, 1].reshape(-1, 1)
        )
        self.reg["blue reg"].fit(
            train_data[:, 2].reshape(-1, 1), reference_data[:, 2].reshape(-1, 1)
        )

    def predict(self, img):
        mod_img = img.copy()

        for strip in mod_img:
            strip[:, 0] = self.reg["red reg"].predict(strip[:, 0].reshape(-1, 1))
            strip[:, 1] = self.reg["green reg"].predict(strip[:, 1].reshape(-1, 1))
            strip[:, 2] = self.reg["blue reg"].predict(strip[:, 2].reshape(-1, 1))
        # for strip in mod_img:
        #   strip = self.reg.predict(strip)

        # return np.clip(mod_img, 0, 1)
        return mod_img
