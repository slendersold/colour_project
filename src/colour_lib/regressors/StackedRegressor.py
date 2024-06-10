import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler


class StackedRegressor:
    def __init__(self, train_data, reference_data, random_state=None):
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        train_data_scaled = self.scaler_x.fit_transform(train_data)
        reference_data_scaled = self.scaler_y.fit_transform(reference_data)

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
            model.fit(train_data_scaled, reference_data_scaled)

        # Генерируем предсказания базовых моделей для обучающих данных метамодели
        meta_features = np.column_stack(
            [model.predict(train_data_scaled) for name, model in self.base_models]
        )

        # Обучаем метамодель
        self.meta_model = MultiOutputRegressor(LinearRegression())
        self.meta_model.fit(meta_features, reference_data_scaled)

    def predict(self, img):
        mod_img = img.copy()

        for strip in mod_img:
            strip_scaled = self.scaler_x.transform(strip)

            # Генерируем предсказания базовых моделей для текущей полосы
            meta_features = np.column_stack(
                [model.predict(strip_scaled) for name, model in self.base_models]
            )

            # Предсказываем результаты метамоделью
            pred_scaled = self.meta_model.predict(meta_features)
            pred = self.scaler_y.inverse_transform(pred_scaled)

            strip[:, :] = np.clip(pred, 0, 1)

        return mod_img


# Пример использования:
# train_data = np.array([[1, 1, 1], [0.5, 0.5, 0.5], ...])
# reference_data = np.array([[1, 0, 0], [0.5, 0.25, 0.25], ...])
# img = np.random.rand(100, 100, 3)  # Пример изображения

# random_state = 42
# regressor = StackedRegressor(train_data, reference_data, random_state)
# corrected_img = regressor.predict(img)
