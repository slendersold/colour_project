from ..regressors.AbstractRegressor import AbstractRegressor

# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.cross_decomposition import PLSRegression
from sklearn import linear_model
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor


class VoteRegressor(AbstractRegressor):
    def __init__(self, train_data, reference_data, **kwargs):
        # base_models = [
        #     ("svr", SVR()),
        #     ("rf", PLSRegression(n_components=train_data.shape[-1])),
        # ]
        base_models = [
            ("linear", LinearRegression()),
            (
                "rf",
                RandomForestRegressor(
                    max_depth=kwargs["max_depth"], random_state=kwargs["random_state"]
                ),
            ),
            ("lasso", linear_model.Lasso(alpha=kwargs["alpha"])),
        ]
        self.reg = MultiOutputRegressor(
            VotingRegressor(estimators=base_models), n_jobs=-1
        )
        self.reg.fit(train_data, reference_data)

    def predict(self, img):
        return super().predict(img)
