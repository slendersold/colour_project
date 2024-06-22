import numpy as np
from ..regressors.AbstractRegressor import AbstractRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR


class StackedRegressor(AbstractRegressor):
    def __init__(self, train_data, reference_data, random_state=None):
        # Базовые модели
        self.base_models = [
            (
                "rf",
                MultiOutputRegressor(RandomForestRegressor(random_state=random_state)),
            ),
            (
                "gbr",
                MultiOutputRegressor(
                    GradientBoostingRegressor(random_state=random_state)
                ),
            ),
            ("svr", MultiOutputRegressor(SVR())),
        ]

        # Обучаем базовые модели
        for name, model in self.base_models:
            model.fit(train_data, reference_data)

        # Генерируем предсказания базовых моделей для обучающих данных метамодели
        meta_features = np.hstack(
            [model.predict(train_data) for name, model in self.base_models]
        )

        # Обучаем метамодель с регуляризацией
        self.meta_model = MultiOutputRegressor(Ridge(alpha=1.0))
        self.meta_model.fit(meta_features, reference_data)

        # Извлекаем коэффициенты метамодели
        self.meta_model_coefs = [
            estimator.coef_ for estimator in self.meta_model.estimators_
        ]

    def predict(self, img):
        mod_img = img.copy()

        for strip in mod_img:
            # Генерируем предсказания базовых моделей для текущей полосы
            meta_features = np.hstack(
                [model.predict(strip) for name, model in self.base_models]
            )

            # Предсказываем результаты метамоделью
            pred = self.meta_model.predict(meta_features)
            strip[:, :] = np.clip(pred, 0, 1)

        return mod_img
