import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params = {
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                },
                "Decision Tree": {
                    "max_depth": [None, 10, 20],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.1, 0.2],
                    "n_estimators": [100, 200],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 6, 10],
                    "n_estimators": [100, 200],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [100, 200],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.01, 0.1, 0.5],
                    "n_estimators": [50, 100],
                },
            }

            # model_report: dict = evaluate_models(
            #     X_train, y_train, X_test, y_test, models
            # )

            model_report, best_estimators = evaluate_models(
    X_train, y_train, X_test, y_test, models, params
)

            ## to get best model score from dict
            # best_model_score = max(sorted(model_report.values()))

            # ## to get best model name from dict
            # best_model_name = list(model_report.keys())[
            #     list(model_report.values()).index(best_model_score)
            # ]
            # best_model = models[best_model_name]

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = best_estimators[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(
                f"Best model found: {best_model_name} with r2 score: {best_model_score}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e, sys)
