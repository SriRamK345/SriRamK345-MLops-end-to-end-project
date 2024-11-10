import logging
import pandas as pd
from zenml import step
from zenml.client import Client

from model.evaluation import RMSE,MSE,R2
from sklearn.base import RegressorMixin

from typing_extensions import Annotated 
from typing import Tuple

import mlflow
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker="mlflow_tracker_1")
def evaluation_data(model:RegressorMixin,
        X_test:pd.DataFrame,
        y_test:pd.DataFrame) -> tuple[
            Annotated[float,"mse"],
            Annotated[float,"r2"],
            Annotated[float,"rmse"]]:

    """
    Evaluate the model on the test data.
    """
    try:
        prediction = model.predict(X_test)

        mse = MSE().calculate_score(y_test,prediction)
        mlflow.log_metric("mse",mse)
        r2 = R2().calculate_score(y_test,prediction)
        mlflow.log_metric("r2",r2)
        rmse = RMSE().calculate_score(y_test,prediction)
        mlflow.log_metric("rmse",rmse)

        return mse,r2,rmse

    except Exception as e:
        logging.error(e)
        raise e
        