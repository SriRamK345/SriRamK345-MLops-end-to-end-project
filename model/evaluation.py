import logging
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
class Evaluation(ABC):
    """
    Abstract class for model evaluation.
    """
    @abstractmethod
    def calculate_score(self, y_true:np.array, y_pred:np.array):
        """
        Calculate the evaluation score for the given true and predicted values.

        Args:
            y_true (np.array): True values.
            y_pred (np.array): Predicted values.

        Returns:
            float: Evaluation score.
        """
        pass

class MSE(Evaluation):
    """
    Mean Squared Error (MSE) evaluation metric.
    """
    def calculate_score(self, y_true:np.array, y_pred:np.array):
        try:
            logging.info("Calculating MSE")
            mse= mean_squared_error(y_true, y_pred)
            logging.info("MSE calculated:{}".format(mse))
            return mse
        except ImportError:
            logging.error("Error in calculating MSE.{}".format(e))
            raise e

class R2(Evaluation):
    """
    R-squared evaluation metric.
    """
    def calculate_score(self, y_true:np.array, y_pred:np.array):
        try:
            logging.info("Calculating R2")
            r2= r2_score(y_true, y_pred)
            logging.info("R2 calculated:{}".format(r2))
            return r2
        except ImportError:
            logging.error("Error in calculating R2.{}".format(e))
            raise e
    
class RMSE(Evaluation):
    """
    Root Mean Squared Error (RMSE) evaluation metric.
    """
    def calculate_score(self, y_true:np.array, y_pred:np.array):
        try:
            logging.info("Calculating RMSE")
            rmse= mean_squared_error(y_true, y_pred)**0.5
            logging.info("RMSE calculated:{}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in calculating RMSE.{}".format(e))
            raise e
       