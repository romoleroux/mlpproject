import os
import sys

import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.exception import CustomException

## Grid Search
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        best_estimators = {}

        for model_name, model in models.items():
            param_grid = params.get(model_name, {})

            if param_grid:
                gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring="r2")
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            
            else:
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)
                test_model_score = r2_score(y_test, y_test_pred)
                report[model_name] = test_model_score
            y_test_pred = best_model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[model_name] = test_model_score
            best_estimators[model_name] = best_model

        return report, best_estimators
    
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
