import logging
import pandas as pd

from model.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin

#from .config import ModelNameConfig
from steps.config import ModelNameConfig

from zenml import step
from zenml.client import Client
from zenml import step

import mlflow

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker="mlflow_tracker_1")
def model_train(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig = ModelNameConfig()) -> RegressorMixin:

    try:
        model = None
        if config.model_name == "LinearRegressionModel":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError("Model {} not found".format(config.model_name))
    except Exception as e:
        logging.error("Error in training model: {}".format(e))
        raise e