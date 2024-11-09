import logging
import pandas as pd
from zenml import step


@step
def model_train(df: pd.DataFrame) -> None:
    """
    Cleaning data from a data path
    """
    #logging.info("Cleaning data from %s", self.data_path)
    #return pd.read_csv(self.data_path)
    pass